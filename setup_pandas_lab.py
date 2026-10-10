"""Builds the practice dataset for the pandas/NumPy 5-day guide.

Usage:  python setup_pandas_lab.py      (creates ./lab/ in the current folder)

Creates
  lab/data/curves/sofr_ois_YYYY-MM-DD.csv   one end-of-day SOFR OIS curve per business day
  lab/data/positions.csv                     PV01 by tenor for a small rates book
  lab/data/pnl_reported.csv                  daily P&L reported by the risk system
  lab/ref/business_days.txt, lab/ref/tenors.txt
"""
import datetime as dt
import os
import random
import shutil

random.seed(42)

TENORS = ["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
BASE = [4.31, 4.30, 4.25, 4.12, 3.95, 3.72, 3.63, 3.60, 3.68, 3.82, 4.05, 4.01]

shutil.rmtree("lab", ignore_errors=True)
for d in ("lab/data/curves", "lab/ref"):
    os.makedirs(d)

days, d = [], dt.date(2026, 8, 3)
while d <= dt.date(2026, 9, 11):
    if d.weekday() < 5:
        days.append(d)
    d += dt.timedelta(days=1)

with open("lab/ref/business_days.txt", "w") as f:
    f.write("\n".join(x.isoformat() for x in days) + "\n")
with open("lab/ref/tenors.txt", "w") as f:
    f.write("\n".join(TENORS) + "\n")

rates = BASE[:]
true_curves = {}
stale_val = None
for d in days:
    ds = d.isoformat()
    common = random.gauss(0, 0.02)
    rates = [r + common + random.gauss(0, 0.005) for r in rates]
    true_curves[ds] = rates[:]
    if ds == "2026-08-21":                       # whole file missing
        continue
    rows = []
    for t, r in zip(TENORS, rates):
        val, snap, src = round(r, 4), "16:00", "BBG"
        if ds == "2026-08-12" and t == "7Y":
            continue                              # missing tenor
        if t == "20Y" and "2026-08-24" <= ds <= "2026-09-01":
            stale_val = stale_val or val
            val = stale_val                       # stale quote
        if ds == "2026-09-03" and t == "10Y":
            val = round(val + 0.45, 4)            # bad print
        if ds == "2026-08-27" and t == "1Y":
            val = round(val / 100, 6)             # unit error
        if ds == "2026-09-10" and t == "2Y":
            snap = "21:00"                        # wrong snapshot
        sval = "" if (ds == "2026-08-18" and t == "3Y") else str(val)   # blank value
        rows.append(f"{ds},{t},{sval},{src},{snap}")
        if ds == "2026-09-08" and t == "5Y":      # conflicting duplicate
            rows.append(f"{ds},{t},{round(val + 0.07, 4)},BGN,16:00")
    with open(f"lab/data/curves/sofr_ois_{ds}.csv", "w") as f:
        f.write("date,tenor,rate_pct,source,snapshot_time\n" + "\n".join(rows) + "\n")

# A small rates book: PV01 in USD per 1bp rise in each tenor's rate
pv01 = {"ON": 0, "1M": -500, "3M": 1200, "6M": -800, "1Y": 2500, "2Y": -9000,
        "3Y": 4000, "5Y": 15000, "7Y": -6000, "10Y": -12000, "20Y": 3000, "30Y": 5000}
with open("lab/data/positions.csv", "w") as f:
    f.write("tenor,pv01_usd\n" + "\n".join(f"{t},{pv01[t]}" for t in TENORS) + "\n")

# Reported P&L from the "risk system", based on the TRUE (clean) curves plus small noise.
# Two days carry a booking error, so explained P&L will not match.
lines = []
for prev, cur in zip(days[:-1], days[1:]):
    p, c = true_curves[prev.isoformat()], true_curves[cur.isoformat()]
    pnl = sum(pv01[t] * (c[i] - p[i]) * 100 for i, t in enumerate(TENORS))
    pnl += random.gauss(0, 300)
    if cur.isoformat() == "2026-08-07":
        pnl += 40000                              # booking error
    if cur.isoformat() == "2026-09-09":
        pnl -= 25000                              # booking error
    lines.append(f"{cur.isoformat()},{round(pnl, 2)}")
with open("lab/data/pnl_reported.csv", "w") as f:
    f.write("date,pnl_usd\n" + "\n".join(lines) + "\n")

print("Lab ready in ./lab")
