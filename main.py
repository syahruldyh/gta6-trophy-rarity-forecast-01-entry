#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster.

Reads every *.in case file from $MAKEFUN_INPUT_DIR (default /input) and
writes one <id>.out per case to $MAKEFUN_OUTPUT_DIR (default /output).
Each output holds, per trophy, five quantiles (5/25/50/75/95) of the
trophy's earn rate (percent) at the case target time.

Method: dedupe stale tracker snapshots, work on the logit scale, take
the last observed rate as the centre (a recency-weighted trend fit is
layered on in forecast_case), and spread with a random-walk term plus a
player-base jump term. Standard library only, fully deterministic.
"""

import glob
import json
import math
import os
from datetime import datetime, timedelta, timezone

EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_from_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def logit(rate_pct):
    p = min(99.99, max(0.01, rate_pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lv):
    if lv >= 0.0:
        return 100.0 / (1.0 + math.exp(-lv))
    e = math.exp(lv)
    return 100.0 * e / (1.0 + e)


def kept_reads(history):
    """Drop stale snapshots: a read identical to the previous kept read
    in both players_tracked and the full rates list carries no signal."""
    kept = []
    for read in history:
        if kept:
            prev = kept[-1]
            if (read.get("players_tracked") == prev.get("players_tracked")
                    and read.get("rates") == prev.get("rates")):
                continue
        kept.append(read)
    return kept


def jump_sigma(history):
    """Player-base jump risk: a frozen tracker means the next real read
    can lurch; a fast-growing base also moves rates; else calm."""
    players = [r.get("players_tracked") for r in history]
    if not players:
        return 0.15
    if all(p == players[0] for p in players):
        return 0.55
    last_at = parse_time(history[-1]["at"])
    last_p = players[-1]
    ref_p = players[0]
    for read in reversed(history):
        if parse_time(read["at"]) <= last_at - timedelta(hours=24):
            ref_p = read.get("players_tracked")
            break
    if ref_p and last_p and last_p > 1.25 * ref_p:
        return 0.30
    return 0.15


def fit_trend(points):
    """Recency-weighted least squares of logit(rate) on x = ln(1 + t).

    Weights halve every 36 hours looking back from the last point.
    Returns (predict, sigma_fit_raw, x_last) where predict maps an x to
    the fitted logit value and sigma_fit_raw is the weighted residual SD.
    """
    t_last = points[-1][0]
    xs = [math.log(1.0 + t) for t, _ in points]
    ys = [logit(r) for _, r in points]
    ws = [0.5 ** ((t_last - t) / 36.0) for t, _ in points]
    sw = sum(ws)
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx > 0.0:
        sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
        slope = sxy / sxx
    else:
        slope = 0.0

    def predict(x):
        return ybar + slope * (x - xbar)

    var = sum(w * (y - predict(x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return predict, math.sqrt(var), xs[-1]


def forecast_case(case):
    history = case.get("history", [])
    kept = kept_reads(history)
    trophies = case.get("trophies", [])
    horizon_min = float(case.get("horizon_minutes", 0.0))
    target_dt = parse_time(case["target"])
    x_target = math.log(1.0 + hours_from_epoch(target_dt))

    sigma_rw = 0.5 * math.sqrt(horizon_min / 1440.0)
    sigma_jump = jump_sigma(history)

    rows = []
    for j, trophy in enumerate(trophies):
        points = []  # (t_hours, rate) for kept reads with a non-null rate
        for read in kept:
            rates = read.get("rates") or []
            rate = rates[j] if j < len(rates) else None
            if rate is not None:
                t = hours_from_epoch(parse_time(read["at"]))
                points.append((t, float(rate)))

        sigma_fit = 0.0
        if points:
            centre = logit(points[-1][1])
            if len(points) >= 3:
                predict, resid_sd, x_last = fit_trend(points)
                delta = 0.7 * (predict(x_target) - predict(x_last))
                delta = min(3.2, max(-1.5, delta))
                centre += delta
                grow = 1.0 + 0.5 * (x_target - x_last) / max(1.0, x_last)
                sigma_fit = min(1.2, resid_sd * grow)
        else:
            centre = logit(0.05)

        sigma = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_jump ** 2)
        sigma = min(1.75, max(0.30, sigma))

        qs = []
        for z in Z:
            q = round(inv_logit(centre + z * sigma), 4)
            q = min(100.0, max(0.0, q))
            if qs and q < qs[-1]:
                q = qs[-1]
            qs.append(q)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, str(case["id"]) + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
