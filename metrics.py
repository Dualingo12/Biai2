import numpy as np


def profit_factor(trades):
    if len(trades) == 0:
        return 0.0
    gains = trades[trades > 0].sum()
    losses = -trades[trades < 0].sum()
    return gains / losses if losses != 0 else 0.0


def max_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_metrics(trades, equity_curve=None):
    if len(trades) == 0:
        metrics = {
            "trades": 0,
            "winrate": 0.0,
            "avg_return": 0.0,
            "profit_factor": 0.0,
        }
        if equity_curve is not None:
            metrics["max_drawdown"] = 0.0
        return metrics

    trades = np.array(trades, dtype=np.float32)
    metrics = {
        "trades": int(len(trades)),
        "winrate": float((trades > 0).mean()),
        "avg_return": float(trades.mean()),
        "profit_factor": float(profit_factor(trades)),
    }
    if equity_curve is not None:
        metrics["max_drawdown"] = max_drawdown(equity_curve)
    return metrics
