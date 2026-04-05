"""ST458 Group J — final submission.

Implements initialise_state(data) and trading_algorithm(new_data, state).

Loads a frozen model (model_linear.npz or model_lgbm_dump.json) and three
JSON config files. Computes features from rolling OHLCV history, scores with
the trained model, and maps scores into risk-controlled positions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
MODEL_LINEAR_NPZ = THIS_DIR / "model_linear.npz"
MODEL_LGBM_JSON = THIS_DIR / "model_lgbm_dump.json"
FEATURE_SPEC_JSON = THIS_DIR / "feature_spec.json"
RISK_JSON = THIS_DIR / "risk_config.json"
DEPLOYMENT_MANIFEST_JSON = THIS_DIR / "deployment_manifest.json"

REQUIRED_COLUMNS = {"date", "symbol", "open", "close", "low", "high", "volume"}


def _require_file(path: Path) -> None:
    """Raise if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path.name}")


def _load_json(path: Path):
    """Load a JSON file after confirming that it exists."""
    _require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num, den):
    """Elementwise divide with zero / inf / nan protection."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = num / np.where(den == 0, np.nan, den)
    out[~np.isfinite(out)] = 0.0
    return out


def _xs_rank(arr):
    """Cross-sectional rank scaled to roughly [-0.5, 0.5]."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    safe = np.where(np.isfinite(arr), arr, 0.0)
    order = np.argsort(safe, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    return ranks / max(len(arr), 1) - 0.5


def _xs_z(arr):
    """Cross-sectional z-score with safe fallback to zeros."""
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if (not np.isfinite(sd)) or sd == 0:
        return np.zeros_like(arr, dtype=float)
    out = (arr - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def _roll_last(x, lb):
    """Return the last lb rows from a 2D history array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] == 0:
        return np.zeros((0, x.shape[1]), dtype=float)
    return x[-min(lb, x.shape[0]):]


def _roll_mean_last(x, lb):
    """Rolling mean over the last lb rows."""
    block = _roll_last(x, lb)
    if block.size == 0:
        return np.zeros(x.shape[1] if np.asarray(x).ndim > 1 else 1, dtype=float)
    out = np.nanmean(block, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _roll_std_last(x, lb):
    """Rolling standard deviation over the last lb rows."""
    block = _roll_last(x, lb)
    if block.size == 0:
        return np.zeros(x.shape[1] if np.asarray(x).ndim > 1 else 1, dtype=float)
    out = np.nanstd(block, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _roll_min_last(x, lb):
    """Rolling minimum over the last lb rows."""
    block = _roll_last(x, lb)
    if block.size == 0:
        return np.zeros(x.shape[1] if np.asarray(x).ndim > 1 else 1, dtype=float)
    out = np.nanmin(block, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _roll_max_last(x, lb):
    """Rolling maximum over the last lb rows."""
    block = _roll_last(x, lb)
    if block.size == 0:
        return np.zeros(x.shape[1] if np.asarray(x).ndim > 1 else 1, dtype=float)
    out = np.nanmax(block, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _ret_from_close(close_hist, lb):
    """Close-to-close return over lb days."""
    close_hist = np.asarray(close_hist, dtype=float)
    if close_hist.shape[0] <= lb:
        return np.zeros(close_hist.shape[1], dtype=float)
    out = _safe_div(close_hist[-1], close_hist[-1 - lb]) - 1.0
    out[~np.isfinite(out)] = 0.0
    return out


def _rolling_semistd_last(ret_hist, lb, positive):
    """Rolling upside or downside semivolatility."""
    block = _roll_last(ret_hist, lb)
    if block.size == 0:
        return np.zeros(ret_hist.shape[1] if np.asarray(ret_hist).ndim > 1 else 1, dtype=float)
    if positive:
        block = np.where(block > 0, block, 0.0)
    else:
        block = np.where(block < 0, block, 0.0)
    out = np.nanstd(block, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def _rolling_beta_last(asset_ret_hist, factor_ret_hist, lb):
    """Rolling beta of each asset versus a scalar factor return history."""
    asset_ret_hist = np.asarray(asset_ret_hist, dtype=float)
    factor_ret_hist = np.asarray(factor_ret_hist, dtype=float).reshape(-1)
    if asset_ret_hist.shape[0] < 2:
        return np.zeros(asset_ret_hist.shape[1], dtype=float)
    block_x = _roll_last(asset_ret_hist, lb)
    block_f = factor_ret_hist[-block_x.shape[0]:]
    f_var = np.nanvar(block_f)
    if (not np.isfinite(f_var)) or f_var == 0:
        return np.zeros(asset_ret_hist.shape[1], dtype=float)
    betas = np.zeros(asset_ret_hist.shape[1], dtype=float)
    f_center = block_f - np.nanmean(block_f)
    for j in range(asset_ret_hist.shape[1]):
        y = block_x[:, j]
        y_center = y - np.nanmean(y)
        cov = np.nanmean(y_center * f_center)
        betas[j] = cov / f_var if np.isfinite(cov) else 0.0
    betas[~np.isfinite(betas)] = 0.0
    return betas


def _peer_index_from_history(ret_hist, window_peer=63):
    """Choose the nearest peer for each asset using rolling return correlation."""
    ret_hist = np.asarray(ret_hist, dtype=float)
    n = ret_hist.shape[1] if ret_hist.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=int)
    if ret_hist.shape[0] < 5:
        return (np.arange(n) + 1) % n
    block = _roll_last(ret_hist, window_peer)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(block, rowvar=False)
    corr = np.nan_to_num(corr, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    np.fill_diagonal(corr, -np.inf)
    peer_idx = np.argmax(corr, axis=1)
    bad = (~np.isfinite(corr.max(axis=1))) | (peer_idx == np.arange(n))
    if np.any(bad):
        peer_idx[bad] = (np.arange(n)[bad] + 1) % n
    return peer_idx.astype(int)


def _cluster_labels_from_history(ret_hist, k=6):
    """Build simple correlation-sorted cluster labels from return history."""
    ret_hist = np.asarray(ret_hist, dtype=float)
    n = ret_hist.shape[1] if ret_hist.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    block = _roll_last(ret_hist, min(63, max(20, ret_hist.shape[0])))
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(block, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    avg_corr = corr.mean(axis=1)
    order = np.argsort(avg_corr)
    buckets = np.array_split(order, min(k, n))
    labels = np.zeros(n, dtype=int)
    for cid, idxs in enumerate(buckets):
        labels[np.asarray(idxs, dtype=int)] = cid
    return labels.astype(int)


def _load_linear_bundle(path: Path):
    """Load an exported linear-like npz bundle."""
    _require_file(path)
    with np.load(path, allow_pickle=True) as z:
        bundle = {k: z[k] for k in z.files}
    bundle["feature_names"] = [str(x) for x in bundle["feature_names"].tolist()]
    bundle["median"] = np.asarray(bundle["median"], dtype=float)
    bundle["mean"] = np.asarray(bundle["mean"], dtype=float)
    bundle["std"] = np.asarray(bundle["std"], dtype=float)
    bundle["coef"] = np.asarray(bundle["coef"], dtype=float).reshape(-1)
    bundle["intercept"] = float(np.asarray(bundle["intercept"], dtype=float).reshape(-1)[0])
    bundle["family"] = str(np.asarray(bundle.get("family", ["ridge"]), dtype=object).reshape(-1)[0])
    bundle["target_col"] = str(np.asarray(bundle.get("target_col", ["target_r1"]), dtype=object).reshape(-1)[0])
    bundle["artifact_format"] = "linear_npz"
    return bundle


def _load_lgbm_bundle(path: Path):
    """Load an exported LightGBM JSON bundle."""
    payload = _load_json(path)
    payload["feature_names"] = [str(x) for x in payload.get("feature_names", [])]
    payload["median"] = np.asarray(payload.get("median", []), dtype=float)
    payload["mean"] = np.asarray(payload.get("mean", []), dtype=float)
    payload["std"] = np.asarray(payload.get("std", []), dtype=float)
    payload["tree_info"] = list(payload.get("tree_info", []))
    payload["family"] = str(payload.get("family", "lgbm"))
    payload["target_col"] = str(payload.get("target_col", "target_r1"))
    payload["artifact_format"] = str(payload.get("artifact_format", "lgbm_json_v1"))
    return payload


def _load_artifacts():
    """Load feature metadata, risk config, manifest, and the model bundle."""
    _require_file(FEATURE_SPEC_JSON)
    _require_file(RISK_JSON)
    _require_file(DEPLOYMENT_MANIFEST_JSON)

    feature_spec = _load_json(FEATURE_SPEC_JSON)
    risk_cfg = _load_json(RISK_JSON)
    deployment_manifest = _load_json(DEPLOYMENT_MANIFEST_JSON)

    artifact_file = str(deployment_manifest.get("artifact_file", ""))
    artifact_format = str(deployment_manifest.get("artifact_format", ""))

    if artifact_format == "linear_npz":
        model_bundle = _load_linear_bundle(THIS_DIR / artifact_file)
    elif artifact_format == "lgbm_json_v1":
        model_bundle = _load_lgbm_bundle(THIS_DIR / artifact_file)
    else:
        raise RuntimeError(f"Unsupported artifact_format in deployment manifest: {artifact_format}")

    expected_features = list(deployment_manifest.get("selected_feature_names", []))
    if expected_features and list(model_bundle["feature_names"]) != expected_features:
        raise RuntimeError("Deployment manifest selected_feature_names does not match the exported model bundle.")
    if len(model_bundle["feature_names"]) != int(deployment_manifest.get("selected_feature_count", len(model_bundle["feature_names"]))):
        raise RuntimeError("Deployment manifest selected_feature_count does not match the exported model bundle.")
    if str(deployment_manifest.get("family", model_bundle["family"])) != str(model_bundle["family"]):
        raise RuntimeError("Deployment manifest family does not match the exported model bundle.")
    return feature_spec, risk_cfg, deployment_manifest, model_bundle


def _prepare_model_matrix(model_bundle, X_df):
    """Reindex, impute, and standardise live features exactly like the export path."""
    feat_names = list(model_bundle["feature_names"])
    X = X_df.reindex(columns=feat_names).to_numpy(dtype=float)
    median = np.asarray(model_bundle["median"], dtype=float)
    mean = np.asarray(model_bundle["mean"], dtype=float)
    std = np.asarray(model_bundle["std"], dtype=float)

    if X.shape[1] != len(feat_names):
        raise RuntimeError("Live feature matrix width does not match exported model bundle.")

    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(median, np.where(bad)[1])

    denom = np.where(std == 0, 1.0, std)
    return (X - mean) / denom


def _apply_linear_bundle(model_bundle, X_df):
    """Score a feature matrix with the exported linear-like bundle."""
    Xs = _prepare_model_matrix(model_bundle, X_df)
    coef = np.asarray(model_bundle["coef"], dtype=float)
    intercept = float(model_bundle["intercept"])
    score = Xs @ coef + intercept
    score = np.asarray(score, dtype=float).reshape(-1)
    score[~np.isfinite(score)] = 0.0
    return score


def _lgbm_go_left(decision_type, value, threshold):
    """Decide whether a LightGBM tree split should go left."""
    dt = str(decision_type)
    if "==" in dt:
        return value == threshold
    return value <= threshold


def _eval_lgbm_tree_node(node, row):
    """Recursively evaluate one LightGBM tree node for a single row."""
    if "leaf_value" in node:
        return float(node["leaf_value"])
    split_feature = int(node["split_feature"])
    threshold = node["threshold"]
    decision_type = node.get("decision_type", "<=")
    default_left = bool(node.get("default_left", True))
    value = float(row[split_feature]) if np.isfinite(row[split_feature]) else np.nan
    if not np.isfinite(value):
        next_node = node["left_child"] if default_left else node["right_child"]
        return _eval_lgbm_tree_node(next_node, row)
    go_left = _lgbm_go_left(decision_type, value, threshold)
    next_node = node["left_child"] if go_left else node["right_child"]
    return _eval_lgbm_tree_node(next_node, row)


def _apply_lgbm_bundle(model_bundle, X_df):
    """Score a feature matrix with the exported pure-Python LightGBM bundle."""
    Xs = _prepare_model_matrix(model_bundle, X_df)
    out = np.zeros(Xs.shape[0], dtype=float)
    for tree in model_bundle.get("tree_info", []):
        tree_structure = tree.get("tree_structure", {})
        for i in range(Xs.shape[0]):
            out[i] += _eval_lgbm_tree_node(tree_structure, Xs[i])
    out[~np.isfinite(out)] = 0.0
    return out


def _apply_model_bundle(model_bundle, X_df):
    """Dispatch to the correct exported-model scoring path."""
    artifact_format = str(model_bundle.get("artifact_format", ""))
    if artifact_format == "linear_npz":
        return _apply_linear_bundle(model_bundle, X_df)
    if artifact_format == "lgbm_json_v1":
        return _apply_lgbm_bundle(model_bundle, X_df)
    raise RuntimeError(f"Unsupported artifact format at score time: {artifact_format}")


def _score_to_target_positions(scores_arr, wealth, current_positions_arr, portfolio_cfg):
    """Convert model scores into target positions under portfolio caps and turnover limits."""
    scores = np.asarray(scores_arr, dtype=float)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    current_positions = np.asarray(current_positions_arr, dtype=float)
    n = scores.shape[0]
    if n == 0 or wealth <= 0:
        return np.zeros_like(current_positions), {"turnover_after_clip": 0.0}

    mapping_method = str(portfolio_cfg.get("mapping_method", "top_bottom"))
    top_n = int(max(2, min(int(portfolio_cfg.get("top_n", 12)), max(2, n // 2))))
    gross = float(min(float(portfolio_cfg.get("gross_exposure", 1.0)), float(portfolio_cfg.get("gross_exposure_cap", 1.5))))
    per_asset_cap = abs(float(portfolio_cfg.get("per_asset_cap", 0.05))) * max(wealth, 1e-12)
    trade_cap = abs(float(portfolio_cfg.get("trade_cap", 0.10))) * max(wealth, 1e-12)
    daily_turnover_cap = abs(float(portfolio_cfg.get("daily_turnover_cap", 0.50))) * max(wealth, 1e-12)
    no_trade_band = abs(float(portfolio_cfg.get("no_trade_band", 0.0))) * max(wealth, 1e-12)

    z = _xs_z(scores)

    if mapping_method == "softmax":
        temp = np.exp(np.clip(z, -8, 8))
        temp = temp / max(temp.sum(), 1e-12)
        w = temp - temp.mean()
    elif mapping_method == "zscore":
        w = np.clip(z, -2.5, 2.5)
    elif mapping_method == "sigmoid":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -8, 8)))
        w = sig - np.mean(sig)
    else:
        order = np.argsort(scores, kind="mergesort")
        w = np.zeros(n, dtype=float)
        short_idx = order[:top_n]
        long_idx = order[-top_n:]
        if len(long_idx):
            w[long_idx] = 1.0 / len(long_idx)
        if len(short_idx):
            w[short_idx] = -1.0 / len(short_idx)

    if portfolio_cfg.get("dollar_neutral", True) or str(portfolio_cfg.get("net_mode", "neutral")) == "neutral":
        w = w - np.mean(w)
    elif str(portfolio_cfg.get("net_mode", "neutral")) == "mild_tilt":
        w = w - 0.75 * np.mean(w)

    gross_norm = np.sum(np.abs(w))
    if gross_norm > 0:
        w = w / gross_norm

    target = wealth * gross * w
    target = np.clip(target, -per_asset_cap, per_asset_cap)

    trades = target - current_positions
    trades[np.abs(trades) < no_trade_band] = 0.0
    trades = np.clip(trades, -trade_cap, trade_cap)

    turnover = float(np.sum(np.abs(trades)))
    if daily_turnover_cap > 0 and turnover > daily_turnover_cap:
        trades *= daily_turnover_cap / max(turnover, 1e-12)

    target_after_clip = current_positions + trades
    return target_after_clip, {"turnover_after_clip": float(np.sum(np.abs(trades)))}


def _append_new_history(state, new_close, new_open, new_low, new_high, new_volume):
    """Append the newest bar into rolling history arrays."""
    close_hist = np.vstack([state["close_hist"], new_close.reshape(1, -1)])
    open_hist = np.vstack([state["open_hist"], new_open.reshape(1, -1)])
    low_hist = np.vstack([state["low_hist"], new_low.reshape(1, -1)])
    high_hist = np.vstack([state["high_hist"], new_high.reshape(1, -1)])
    vol_hist = np.vstack([state["vol_hist"], new_volume.reshape(1, -1)])
    return close_hist, open_hist, low_hist, high_hist, vol_hist


def _build_price_action_features(close_hist, new_close, new_open, new_low, new_high, state):
    """Price-action and candle-shape features."""
    ret_hist = _safe_div(close_hist[1:], close_hist[:-1]) - 1.0 if close_hist.shape[0] >= 2 else np.zeros((0, close_hist.shape[1]), dtype=float)
    ret_1 = ret_hist[-1] if ret_hist.shape[0] else np.zeros_like(new_close, dtype=float)
    ret_1 = np.where(np.isfinite(ret_1), ret_1, 0.0)

    feats = {
        "ret_1": ret_1,
        "ret_2": _ret_from_close(close_hist, 2),
        "ret_3": _ret_from_close(close_hist, 3),
        "ret_5": _ret_from_close(close_hist, 5),
        "ret_10": _ret_from_close(close_hist, 10),
        "ret_20": _ret_from_close(close_hist, 20),
        "ret_40": _ret_from_close(close_hist, 40),
        "ret_63": _ret_from_close(close_hist, 63),
        "oc_ret_0": _safe_div(new_close, new_open) - 1.0,
        "gap_co_1": _safe_div(new_open, state["prev_close"]) - 1.0,
    }

    abs_span = np.abs(new_high - new_low)
    span_safe = np.where(abs_span == 0, np.nan, abs_span)
    feats["range_0"] = _safe_div(new_high - new_low, np.where(new_close == 0, np.nan, new_close))
    feats["close_loc_0"] = _safe_div(new_close - new_low, span_safe)
    feats["body_ratio_0"] = _safe_div(new_close - new_open, span_safe)
    feats["upper_wick_0"] = _safe_div(new_high - np.maximum(new_open, new_close), span_safe)
    feats["lower_wick_0"] = _safe_div(np.minimum(new_open, new_close) - new_low, span_safe)

    for lb in [5, 10, 20, 40]:
        ma = _roll_mean_last(close_hist, lb)
        roll_min = _roll_min_last(close_hist, lb)
        roll_max = _roll_max_last(close_hist, lb)
        denom = np.where((roll_max - roll_min) == 0, np.nan, (roll_max - roll_min))
        feats[f"ma_gap_{lb}"] = _safe_div(new_close, ma) - 1.0
        feats[f"dist_min_{lb}"] = _safe_div(new_close - roll_min, denom)
        feats[f"dist_max_{lb}"] = _safe_div(roll_max - new_close, denom)

    feats["mom_5_20"] = feats["ret_5"] - feats["ret_20"]
    feats["mom_10_40"] = feats["ret_10"] - feats["ret_40"]
    feats["accel_1_5_20"] = feats["ret_1"] - 2.0 * feats["ret_5"] + feats["ret_20"]
    feats["signed_sqrt_absret_1"] = np.sign(feats["ret_1"]) * np.sqrt(np.abs(feats["ret_1"]))
    rolling_vol20 = _roll_std_last(ret_hist, 20)
    big_move = np.abs(feats["ret_1"]) > rolling_vol20
    feats["cond_reversal_1"] = np.where(big_move, -feats["ret_1"], 0.0)
    return feats, ret_hist


def _build_volatility_features(ret_hist):
    """Volatility, semivolatility, and short-horizon return means."""
    feats = {}
    for lb in [5, 10, 20]:
        feats[f"vol_{lb}"] = _roll_std_last(ret_hist, lb)
        feats[f"down_vol_{lb}"] = _rolling_semistd_last(ret_hist, lb, positive=False)
        feats[f"up_vol_{lb}"] = _rolling_semistd_last(ret_hist, lb, positive=True)
        feats[f"ret_mean_{lb}"] = _roll_mean_last(ret_hist, lb)
    return feats


def _build_volume_features(vol_hist, new_volume, base_feats):
    """Volume, liquidity, and flow-proxy features."""
    log_vol_hist = np.log1p(np.where(vol_hist < 0, 0.0, vol_hist))
    log_volume = log_vol_hist[-1]
    prev_vol = vol_hist[-2] if vol_hist.shape[0] >= 2 else np.ones_like(new_volume)
    vol_surp5_denom = np.where(_roll_std_last(log_vol_hist, 5) == 0, np.nan, _roll_std_last(log_vol_hist, 5))
    vol_surp20_denom = np.where(_roll_std_last(log_vol_hist, 20) == 0, np.nan, _roll_std_last(log_vol_hist, 20))

    feats = {
        "log_volume": np.where(np.isfinite(log_volume), log_volume, 0.0),
        "vol_chg_1": _safe_div(new_volume, prev_vol) - 1.0,
        "vol_surprise_5": _safe_div(log_volume - _roll_mean_last(log_vol_hist, 5), vol_surp5_denom),
        "vol_surprise_20": _safe_div(log_volume - _roll_mean_last(log_vol_hist, 20), vol_surp20_denom),
        "volume_vol_20": _roll_std_last(log_vol_hist, 20),
    }
    feats["bigmove_x_volsurprise"] = np.abs(base_feats["ret_1"]) * feats["vol_surprise_20"]
    feats["signedmove_x_volume"] = base_feats["ret_1"] * _xs_z(feats["log_volume"])
    feats["range_x_volume"] = base_feats["range_0"] * _xs_z(feats["log_volume"])
    feats["liq_pressure_proxy"] = _safe_div(base_feats["range_0"], np.where(feats["log_volume"] == 0, np.nan, feats["log_volume"]))
    feats["turnover_pressure"] = (np.abs(base_feats["ret_1"]) + np.abs(base_feats["range_0"])) * np.abs(feats["vol_surprise_20"])
    return feats


def _build_factor_residual_features(ret_hist, base_feats):
    """Market, beta, residual, and dispersion features."""
    if ret_hist.shape[0]:
        market_ret_hist = np.nanmean(ret_hist, axis=1)
        market_ret_1 = float(market_ret_hist[-1])
        market_vol_20_scalar = float(np.nanstd(market_ret_hist[-min(20, len(market_ret_hist)):]))
        beta20 = _rolling_beta_last(ret_hist, market_ret_hist, 20)
        beta60 = _rolling_beta_last(ret_hist, market_ret_hist, 60)
        resid_ret_1 = base_feats["ret_1"] - beta60 * market_ret_1
        resid_hist = ret_hist - market_ret_hist.reshape(-1, 1)
        resid_mom_5 = _roll_mean_last(resid_hist, 5)
        idio_vol_20 = _roll_std_last(resid_hist, 20)
        dispersion_scalar = float(np.nanmean(np.nanstd(_roll_last(ret_hist, 20), axis=1))) if _roll_last(ret_hist, 20).size else 0.0
    else:
        market_ret_1 = 0.0
        market_vol_20_scalar = 0.0
        beta20 = np.zeros_like(base_feats["ret_1"], dtype=float)
        beta60 = np.zeros_like(base_feats["ret_1"], dtype=float)
        resid_ret_1 = np.zeros_like(base_feats["ret_1"], dtype=float)
        resid_mom_5 = np.zeros_like(base_feats["ret_1"], dtype=float)
        idio_vol_20 = np.zeros_like(base_feats["ret_1"], dtype=float)
        dispersion_scalar = 0.0

    return {
        "market_ret_1": np.repeat(market_ret_1, len(base_feats["ret_1"])),
        "market_vol_20": np.repeat(market_vol_20_scalar, len(base_feats["ret_1"])),
        "beta_20": beta20,
        "beta_60": beta60,
        "resid_ret_1": resid_ret_1,
        "resid_mom_5": resid_mom_5,
        "idio_vol_20": idio_vol_20,
        "dispersion_20": np.repeat(dispersion_scalar, len(base_feats["ret_1"])),
    }


def _build_pair_features(close_hist, ret_hist, base_feats, state):
    """Pair-spread and peer-relative features."""
    peer_idx = _peer_index_from_history(ret_hist, int(state.get("peer_window", 63)))
    log_close_hist = np.log(np.clip(close_hist, 1e-12, None))
    log_new_close = log_close_hist[-1]
    pair_spread_current = log_new_close - log_new_close[peer_idx]
    pair_spread_hist = np.vstack([np.asarray(state["pair_spread_hist"], dtype=float), pair_spread_current.reshape(1, -1)])
    spread_window = _roll_last(pair_spread_hist, int(state.get("pair_z_window", 20)))
    spread_mu = np.nanmean(spread_window, axis=0)
    spread_sd = np.nanstd(spread_window, axis=0)
    pair_spread_z = _safe_div(pair_spread_current - spread_mu, np.where(spread_sd == 0, np.nan, spread_sd))
    pair_resid_ret = base_feats["ret_1"] - base_feats["ret_1"][peer_idx]

    pair_spread_mr = -1.0 * pair_spread_z
    feats = {
        "pair_resid_ret": pair_resid_ret,
        "pair_spread_z": pair_spread_z,
        "pair_spread_mr": pair_spread_mr,
        "xs_rank_pair_resid_ret": _xs_rank(pair_resid_ret),
        "xs_z_pair_resid_ret": _xs_z(pair_resid_ret),
        "xs_z_pair_spread_z": _xs_z(pair_spread_z),
    }
    updates = {
        "pair_spread_hist": pair_spread_hist[-int(state["max_history_lookback"]):],
        "peer_idx": peer_idx.astype(int),
    }
    return feats, updates


def _build_cluster_features(base_feats, state):
    """Simple cluster-relative residual features."""
    cluster_labels = np.asarray(state["cluster_labels"], dtype=int)
    cluster_ret_1 = np.zeros_like(base_feats["ret_1"])
    for cid in np.unique(cluster_labels):
        idxs = np.where(cluster_labels == cid)[0]
        if len(idxs):
            cluster_ret_1[idxs] = float(np.nanmean(base_feats["ret_1"][idxs]))
    cluster_resid_1 = base_feats["ret_1"] - cluster_ret_1
    return {
        "cluster_resid_1": cluster_resid_1,
        "xs_z_cluster_resid_1": _xs_z(cluster_resid_1),
    }


def _build_alpha_features(feats):
    """Composite alpha blocks used by the final strategy."""
    alpha_reversal_short = (
        -1.0 * feats.get("xs_z_ret_1", np.zeros_like(feats["ret_1"]))
        -0.4 * feats.get("xs_z_gap_co_1", np.zeros_like(feats["ret_1"]))
        +0.25 * feats.get("xs_z_close_loc_0", np.zeros_like(feats["ret_1"]))
    )
    alpha_momentum_medium = (
        0.7 * feats.get("xs_rank_ret_20", np.zeros_like(feats["ret_1"]))
        +0.6 * feats.get("xs_rank_mom_10_40", np.zeros_like(feats["ret_1"]))
        +0.2 * feats.get("xs_rank_ma_gap_20", np.zeros_like(feats["ret_1"]))
    )
    alpha_factor_resid = (
        0.6 * feats.get("xs_z_resid_ret_1", np.zeros_like(feats["ret_1"]))
        +0.4 * feats.get("xs_z_cluster_resid_1", np.zeros_like(feats["ret_1"]))
        -0.2 * feats.get("xs_z_idio_vol_20", np.zeros_like(feats["ret_1"]))
    )
    alpha_pair_spread = (
        0.8 * feats.get("xs_z_pair_spread_z", np.zeros_like(feats["ret_1"]))
        +0.5 * feats.get("xs_z_pair_resid_ret", np.zeros_like(feats["ret_1"]))
    )
    alpha_vol_conditioned = (
        0.6 * feats.get("xs_rank_ret_5", np.zeros_like(feats["ret_1"]))
        * (1.0 / (1.0 + np.abs(feats.get("idio_vol_20", np.zeros_like(feats["ret_1"])))))
        -0.4 * feats.get("xs_z_down_vol_20", np.zeros_like(feats["ret_1"]))
    )

    return {
        "alpha_reversal_short": alpha_reversal_short,
        "alpha_momentum_medium": alpha_momentum_medium,
        "alpha_factor_resid": alpha_factor_resid,
        "alpha_pair_spread": alpha_pair_spread,
        "alpha_vol_conditioned": alpha_vol_conditioned,
        "alpha_rule_score": (
            alpha_reversal_short
            + alpha_momentum_medium
            + alpha_factor_resid
            + alpha_pair_spread
            + alpha_vol_conditioned
        ) / 5.0,
    }


def _compute_live_features(state, new_close, new_open, new_low, new_high, new_volume):
    """Compute all live-supported features and return an updated history state."""
    close_hist, open_hist, low_hist, high_hist, vol_hist = _append_new_history(
        state, new_close, new_open, new_low, new_high, new_volume
    )

    # Price-action block.
    price_feats, ret_hist = _build_price_action_features(close_hist, new_close, new_open, new_low, new_high, state)

    # Volatility block.
    vol_feats = _build_volatility_features(ret_hist)

    # Volume / liquidity block.
    volume_feats = _build_volume_features(vol_hist, new_volume, {**price_feats, **vol_feats})

    # Factor / residual block.
    factor_feats = _build_factor_residual_features(ret_hist, {**price_feats, **vol_feats, **volume_feats})

    # Pair-relative-value block.
    pair_feats, pair_updates = _build_pair_features(close_hist, ret_hist, {**price_feats, **vol_feats, **volume_feats, **factor_feats}, state)

    # Cluster block.
    cluster_feats = _build_cluster_features({**price_feats, **vol_feats, **volume_feats, **factor_feats, **pair_feats}, state)

    feats = {}
    feats.update(price_feats)
    feats.update(vol_feats)
    feats.update(volume_feats)
    feats.update(factor_feats)
    feats.update(pair_feats)
    feats.update(cluster_feats)

    xs_bases = [
        "ret_1", "ret_5", "ret_10", "ret_20", "gap_co_1",
        "mom_5_20", "mom_10_40", "ma_gap_20",
        "vol_surprise_5", "vol_surprise_20", "range_0", "close_loc_0",
        "resid_ret_1", "idio_vol_20", "down_vol_20",
    ]
    for base in xs_bases:
        if base in feats:
            feats[f"xs_rank_{base}"] = _xs_rank(feats[base])
            feats[f"xs_z_{base}"] = _xs_z(feats[base])

    # Alpha-composite block.
    feats.update(_build_alpha_features(feats))

    feat_df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in feats.items()}, index=state["symbols"])
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    updated_bits = {
        "close_hist": close_hist[-int(state["max_history_lookback"]):],
        "open_hist": open_hist[-int(state["max_history_lookback"]):],
        "low_hist": low_hist[-int(state["max_history_lookback"]):],
        "high_hist": high_hist[-int(state["max_history_lookback"]):],
        "vol_hist": vol_hist[-int(state["max_history_lookback"]):],
        **pair_updates,
    }
    return feat_df, updated_bits


def initialise_state(data):
    """Initialise rolling histories, load artifacts, and build the persistent strategy state."""
    df = data.copy()
    missing_cols = REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Training data missing required columns: {sorted(missing_cols)}")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    if df[["date", "symbol"]].duplicated().any():
        raise ValueError("Training data contains duplicate date-symbol rows.")
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_spec, risk_cfg, deployment_manifest, model_bundle = _load_artifacts()

    symbols = sorted(df["symbol"].unique().tolist())
    expected_symbol_count = int(feature_spec.get("expected_symbol_count", len(symbols)))
    if len(symbols) != expected_symbol_count:
        raise ValueError(f"Training universe size {len(symbols)} does not match exported expectation {expected_symbol_count}.")

    close_wide = df.pivot(index="date", columns="symbol", values="close").reindex(columns=symbols)
    open_wide = df.pivot(index="date", columns="symbol", values="open").reindex(columns=symbols)
    low_wide = df.pivot(index="date", columns="symbol", values="low").reindex(columns=symbols)
    high_wide = df.pivot(index="date", columns="symbol", values="high").reindex(columns=symbols)
    vol_wide = df.pivot(index="date", columns="symbol", values="volume").reindex(columns=symbols)

    max_hist = int(feature_spec.get("max_history_lookback", 80))
    close_hist = close_wide.tail(max_hist).to_numpy(dtype=float)
    open_hist = open_wide.tail(max_hist).to_numpy(dtype=float)
    low_hist = low_wide.tail(max_hist).to_numpy(dtype=float)
    high_hist = high_wide.tail(max_hist).to_numpy(dtype=float)
    vol_hist = vol_wide.tail(max_hist).to_numpy(dtype=float)

    ret_hist = _safe_div(close_hist[1:], close_hist[:-1]) - 1.0 if close_hist.shape[0] >= 2 else np.zeros((0, close_hist.shape[1]), dtype=float)
    peer_window = int(feature_spec.get("peer_window", 63))
    pair_z_window = int(feature_spec.get("pair_z_window", 20))
    cluster_k = int(feature_spec.get("cluster_k", 6))

    peer_idx = _peer_index_from_history(ret_hist, peer_window)
    log_close_hist = np.log(np.clip(close_hist, 1e-12, None))
    pair_spread_hist = log_close_hist - log_close_hist[:, peer_idx]
    cluster_labels = _cluster_labels_from_history(ret_hist, cluster_k)

    selected_features = list(feature_spec.get("selected_features", []))
    live_feature_support = set(feature_spec.get("live_feature_support", []))
    missing_live = [f for f in selected_features if f not in live_feature_support]
    if missing_live:
        raise RuntimeError(f"Selected features are not supported by the live engine: {missing_live}")

    portfolio_cfg = dict(risk_cfg.get("portfolio", {}))
    portfolio_cfg.setdefault("mapping_method", "top_bottom")
    portfolio_cfg.setdefault("gross_exposure", 1.0)
    portfolio_cfg.setdefault("gross_exposure_cap", 1.5)
    portfolio_cfg.setdefault("top_n", 12)
    portfolio_cfg.setdefault("per_asset_cap", 0.05)
    portfolio_cfg.setdefault("trade_cap", 0.10)
    portfolio_cfg.setdefault("daily_turnover_cap", 0.50)
    portfolio_cfg.setdefault("no_trade_band", 0.0)
    portfolio_cfg.setdefault("dollar_neutral", True)
    portfolio_cfg.setdefault("net_mode", "neutral")

    if list(deployment_manifest.get("selected_feature_names", selected_features)) != list(selected_features):
        raise RuntimeError("feature_spec selected_features and deployment_manifest selected_feature_names disagree.")

    return {
        "symbols": symbols,
        "positions": np.zeros(len(symbols), dtype=float),
        "wealth": 1.0,
        "prev_close": close_hist[-1].copy(),
        "close_hist": close_hist,
        "open_hist": open_hist,
        "low_hist": low_hist,
        "high_hist": high_hist,
        "vol_hist": vol_hist,
        "pair_spread_hist": pair_spread_hist[-max_hist:],
        "peer_idx": peer_idx,
        "cluster_labels": cluster_labels,
        "selected_features": selected_features,
        "max_history_lookback": max_hist,
        "peer_window": peer_window,
        "pair_z_window": pair_z_window,
        "cluster_k": cluster_k,
        "portfolio_cfg": portfolio_cfg,
        "cost_rate": float(risk_cfg.get("cost_rate", 0.0005)),
        "target_daily_vol": float(risk_cfg.get("target_daily_vol", 0.012)),
        "feature_spec": feature_spec,
        "risk_cfg": risk_cfg,
        "deployment_manifest": deployment_manifest,
        "model_bundle": model_bundle,
        "day_count": 0,
    }


def trading_algorithm(new_data, state):
    """Score the new day, map scores into target positions, and return trades plus updated state."""
    day = new_data.copy()
    missing_cols = REQUIRED_COLUMNS.difference(day.columns)
    if missing_cols:
        raise ValueError(f"new_data missing required columns: {sorted(missing_cols)}")

    day["date"] = pd.to_datetime(day["date"])
    if day["date"].nunique() != 1:
        raise ValueError("new_data must contain exactly one trading date.")
    if day["symbol"].duplicated().any():
        raise ValueError("new_data contains duplicate symbols.")
    day["symbol"] = day["symbol"].astype(str)

    symbols = list(state["symbols"])
    sym_rank = {s: i for i, s in enumerate(symbols)}
    day["_r"] = day["symbol"].map(sym_rank)
    if day["_r"].isna().any():
        missing = sorted(day.loc[day["_r"].isna(), "symbol"].unique().tolist())
        raise ValueError(f"Encountered unknown symbols in new_data: {missing}")
    day = day.sort_values("_r").drop(columns=["_r"]).reset_index(drop=True)

    new_close = day["close"].to_numpy(dtype=float)
    new_open = day["open"].to_numpy(dtype=float)
    new_low = day["low"].to_numpy(dtype=float)
    new_high = day["high"].to_numpy(dtype=float)
    new_volume = day["volume"].to_numpy(dtype=float)

    prev_close = np.asarray(state["prev_close"], dtype=float)
    marked_positions = np.asarray(state["positions"], dtype=float)
    r1d = _safe_div(new_close, prev_close) - 1.0
    marked_positions = marked_positions * (1.0 + r1d)
    state["wealth"] = max(0.0, float(state["wealth"] + np.sum(np.asarray(state["positions"], dtype=float) * r1d)))

    feat_df, updated_bits = _compute_live_features(state, new_close, new_open, new_low, new_high, new_volume)
    selected_features = list(state["selected_features"])
    missing_live = [f for f in selected_features if f not in feat_df.columns]
    if missing_live:
        raise RuntimeError(f"Live feature computation failed to produce: {missing_live}")

    raw_score = _apply_model_bundle(state["model_bundle"], feat_df.loc[:, selected_features])
    target_positions, _ = _score_to_target_positions(raw_score, state["wealth"], marked_positions, state["portfolio_cfg"])

    trades = target_positions - marked_positions
    state["positions"] = np.asarray(target_positions, dtype=float)
    state["prev_close"] = np.asarray(new_close, dtype=float)
    state["day_count"] = int(state["day_count"]) + 1
    for key, value in updated_bits.items():
        state[key] = value

    trade_series = pd.Series(trades, index=symbols, dtype=float)
    return trade_series, state
