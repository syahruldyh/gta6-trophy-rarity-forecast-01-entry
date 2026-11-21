# GTA VI Trophy Rarity Forecast

Forecasts, for each of GTA VI's 55 trophies, the distribution of its earn
rate (percent of tracked players who have earned it) at a future target
time. The program (`main.py`, Python 3, standard library only) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR`, each holding the 5/25/50/75/95 quantiles of every
trophy's earn rate at the target.

## Method

- **Snapshot dedupe.** Tracker history often repeats a stale snapshot.
  Reads whose `players_tracked` and full rates list are identical to the
  previous kept read are dropped before fitting.
- **Logit-scale trend.** Earn rates are mapped through a logit transform
  (clamped to 0.01–99.99%). With at least 3 usable snapshots, a
  recency-weighted least-squares line (weights halving every 36 h) is fit
  against `ln(1 + t_hours)`. The forecast centre anchors on the last
  observed logit and adds a damped (×0.7, clamped) share of the fitted
  trend out to the target; with fewer snapshots the last rate is held.
- **Three-part calibrated spread.** The predictive sigma on the logit
  scale combines a random-walk term growing with the horizon
  (`0.5·sqrt(days)`), a fit-uncertainty term from the weighted residual
  SD inflated by the extrapolation distance, and a player-base jump term
  that widens when the tracker base is frozen or growing fast. Sigma is
  floored at 0.30 and capped at 1.75; quantiles use the standard normal
  z-values for the five levels.

## Competition

https://make.fun/competitions/gta6-trophy-rarity-forecast-01

makefun anchor d3ff87be407dbe82b5ae7c568d5afbaeda186e5d round 34151294
