# pandas and NumPy in 5 Days: Daily Practice Guide

This guide prepares you for the hands-on data work in a quantitative data science interview: loading messy market data, checking it, reshaping it, analysing it as a time series, and explaining what you found. It is written for someone who already uses Python and pandas at work, and wants to be fast, idiomatic and confident under interview pressure.

Every exercise uses one realistic dataset: six weeks of daily SOFR OIS curves from a market-data feed, a small rates book's PV01 by tenor, and the daily P&L reported by a risk system. **The data contains deliberate problems, and so does the P&L.** By Day 5 you will find every one of them in Python and explain which P&L breaks come from bad data and which come from somewhere else.

## How to use this guide

- **Type every line yourself** in a Jupyter notebook. Don't copy and paste; the aim is fluency.
- **Predict the output before you run a cell.** When you're wrong, work out why before moving on.
- **Answers are hidden.** Click "Answer" only after a genuine attempt. In plain-text view the answers show directly, so read this file in VS Code's preview (Ctrl+Shift+V) or on GitHub.
- **Each day opens with a warm-up** on the previous day and closes with a **mini-lab**, a timed drill, and a checklist of interview questions.
- Use one notebook per day: `day1.ipynb`, `day2.ipynb`, and so on.

| Day | Topic | Mini-lab |
|---|---|---|
| 1 | NumPy: arrays, vectorisation, broadcasting | Discount factors and a no-arbitrage check |
| 2 | pandas foundations: load, inspect, select, clean | Load six weeks of files and profile them |
| 3 | groupby, reshape, merge: data-quality checks | Find every structural data problem |
| 4 | Time series: diff, shift, rolling, anomaly detection | Build a data-health monitor |
| 5 | Analysis: PCA, regression, P&L attribution, production code | Root-cause every P&L break |

---

## One-off setup (about 20 minutes, before Day 1)

### 1. Python and libraries

You need Python 3.10 or later. In a terminal:

```bash
python3 --version
python3 -m pip install --upgrade pandas numpy matplotlib jupyter
python3 -c "import pandas, numpy; print(pandas.__version__, numpy.__version__)"
```

On Windows use `python` instead of `python3` if that's the command that works. If `pip install` complains about an "externally managed environment", create a virtual environment first:

```bash
python3 -m venv ~/pdcourse-env
source ~/pdcourse-env/bin/activate        # Windows: ~/pdcourse-env/Scripts/activate
python3 -m pip install pandas numpy matplotlib jupyter
```

This guide was tested with **pandas 3.0 and NumPy 2.5**. Everything also works on pandas 2.2 or later; where behaviour differs, the guide says so.

### 2. Jupyter in VS Code

Install the **Python** and **Jupyter** extensions in VS Code. Create a file called `day1.ipynb`, open it, and choose your Python interpreter when asked (top right, "Select Kernel"). **Shift+Enter** runs a cell and moves to the next.

Use a personal laptop rather than a bank-managed machine.

### 3. Create the dataset

```bash
mkdir -p ~/pandas-course
cd ~/pandas-course
# put setup_pandas_lab.py here, then:
python3 setup_pandas_lab.py
```

You should see "Lab ready in ./lab". Keep your notebooks in `~/pandas-course`, so the relative path `lab/...` works in every exercise. Re-run the setup script at any time to reset the data.

```
~/pandas-course/
├── setup_pandas_lab.py
├── day1.ipynb ... day5.ipynb       (yours)
└── lab/
    ├── data/curves/sofr_ois_YYYY-MM-DD.csv    one curve per business day
    ├── data/positions.csv                      PV01 by tenor for a rates book
    ├── data/pnl_reported.csv                   daily P&L from the risk system
    └── ref/business_days.txt, ref/tenors.txt
```

### 4. Know the data

Each curve file:

```
date,tenor,rate_pct,source,snapshot_time
2026-09-03,ON,4.3753,BBG,16:00
2026-09-03,1M,4.3186,BBG,16:00
...
```

- `tenor`: one of 12 maturities, ON (overnight) to 30Y.
- `rate_pct`: the rate in percent, so 4.3753 means 4.3753%. 1 basis point (bp) is 0.01%.
- `source`: BBG (Bloomberg), or another vendor code.
- `snapshot_time`: when the price was captured; it should be 16:00 London.

`positions.csv` gives the book's **PV01** per tenor: the change in the book's value, in USD, when that tenor's rate rises by 1bp. So the P&L explained by curve moves on a day is the sum over tenors of PV01 × (rate change in bp). `pnl_reported.csv` is what the risk system says the book made each day.

---
# Day 1: NumPy

**Goal:** think in whole arrays instead of loops. Create and index arrays, use boolean masks, aggregate along an axis, broadcast, handle NaN, simulate, and do basic linear algebra. pandas is built on NumPy, so everything today makes Days 2–5 faster.
**Time:** about 2 hours.

Start `day1.ipynb` with:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)   # tidy output: 4 decimals, no 1e-05 notation
```

## 1.1 Arrays (15 min)

A NumPy array is a block of numbers that all share one type, stored contiguously in memory. That is what makes it fast.

```python
rates = np.array([4.3753, 4.3186, 4.3001, 4.1763, 3.9955, 3.7731,
                  3.7288, 3.6285, 3.7132, 4.3081, 4.1039, 4.0525])
rates.shape, rates.ndim, rates.dtype, rates.size
```

```python
np.arange(0, 10, 2)          # like range(): 0, 2, 4, 6, 8
np.linspace(0, 30, 7)        # 7 evenly spaced points from 0 to 30, inclusive
np.zeros(3), np.ones(3), np.full(3, 4.25)
np.zeros((2, 3))             # a 2-row, 3-column matrix
```

The maturities, in years, of our 12 tenors:

```python
tenors = np.array(["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"])
mats = np.array([1/360, 1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
```

Reading a column straight from a CSV file with NumPy (normally you'd use pandas, but it's worth knowing):

```python
r = np.genfromtxt("lab/data/curves/sofr_ois_2026-09-03.csv",
                  delimiter=",", skip_header=1, usecols=2)
r
```

That's the 3 September curve, the same numbers as `rates`.

## 1.2 Vectorisation (15 min)

Arithmetic on arrays applies to every element at once, with no Python loop:

```python
rates * 100              # percent to basis points
rates / 100              # percent to decimal
rates - rates.mean()     # deviation from the average
np.exp(-rates / 100 * mats)   # continuously compounded discount factors
```

**Why this matters: speed.** Compare a loop with the vectorised version on a million numbers:

```python
big = np.random.default_rng(0).normal(4, 0.5, 1_000_000)

def loop_bp(x):
    out = []
    for v in x:
        out.append(v * 100)
    return out

%timeit loop_bp(big)
%timeit big * 100
```

`%timeit` is a notebook command that runs a line many times and reports the average. The vectorised version is typically around 100 times faster. In an interview, writing a Python `for` loop over a large array or DataFrame is a signal you don't know the library; reaching for vectorised operations first is expected.

**Element-wise functions** ("ufuncs") you'll use constantly: `np.exp`, `np.log`, `np.sqrt`, `np.abs`, `np.maximum(a, b)`, `np.clip(a, lo, hi)`, `np.round(a, 2)`.

## 1.3 Indexing, slicing and boolean masks (20 min)

```python
rates[0]             # first element
rates[-1]            # last
rates[2:5]           # elements 2, 3, 4 (end is exclusive)
rates[::-1]          # reversed
rates[[0, 5, 9]]     # "fancy indexing": a list of positions → ON, 2Y, 10Y
```

**Boolean masks** are the heart of NumPy and pandas filtering:

```python
mask = rates > 4.2
mask                         # an array of True/False
rates[mask]                  # only the values where mask is True
tenors[mask]                 # the same positions in another array
tenors[(rates > 3.7) & (rates < 3.8)]    # & means and, | means or, ~ means not
mask.sum()                   # True counts as 1: how many rates are above 4.2%
mask.any(), mask.all()
```

Use `&`, `|` and `~` with arrays, never `and`, `or`, `not`, and **always put brackets around each condition**: `(a > 1) & (b < 2)`. Without them, Python's operator precedence evaluates `1 & b` first and you get an error or a wrong answer.

`np.where` is a vectorised if/else:

```python
np.where(rates > 4.2, "high", "low")
np.where(rates < 0.5, np.nan, rates)     # replace implausible values with NaN
```

`argmax` and `argmin` give the *position* of the extreme, which you then use to look up a label:

```python
tenors[rates.argmax()], rates.max()
tenors[rates.argmin()], rates.min()
```

Look at the 3 September curve: 7Y is 3.71%, 10Y is 4.31% and 20Y is 4.10%. A 60bp jump between 7Y and 10Y, with 10Y higher than 20Y, is a strange shape. Keep that in mind for the mini-lab.

### Practice 1

1. Print the tenors whose rate is within 5bp of the curve's average.
2. Replace every rate above 4.3% with 4.3 (a "cap"), without changing `rates` itself.
3. Find the tenor with the biggest gap to its neighbour on the right. (Hint: `np.diff`.)

<details><summary>Answer</summary>

```python
tenors[np.abs(rates - rates.mean()) < 0.05]
capped = np.minimum(rates, 4.3)        # or np.clip(rates, None, 4.3)
gaps = np.diff(rates)                  # rates[1:] - rates[:-1], length 11
i = np.abs(gaps).argmax()
tenors[i], tenors[i + 1], gaps[i] * 100     # 7Y → 10Y, about +59.5bp
```
</details>

## 1.4 2-D arrays and `axis` (20 min)

Load the first five August curves into a matrix: one row per day, one column per tenor.

```python
dates = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
M = np.vstack([
    np.genfromtxt(f"lab/data/curves/sofr_ois_{d}.csv", delimiter=",", skip_header=1, usecols=2)
    for d in dates
])
M.shape          # (5, 12)
```

Indexing in two dimensions is `[rows, columns]`:

```python
M[0]             # first day, all tenors
M[:, 9]          # all days, 10Y (column 9)
M[-1, -1]        # last day, 30Y
M[1:3, :4]       # days 2–3, first four tenors
```

**`axis` tells an aggregation which direction to collapse.** `axis=0` collapses the rows, giving one result per column; `axis=1` collapses the columns, giving one result per row.

```python
M.mean(axis=0)     # average rate per tenor over 5 days → 12 numbers
M.mean(axis=1)     # average rate per day across tenors → 5 numbers
M.std(axis=0)      # volatility of each tenor's level
M.max()            # no axis: the single maximum
```

Day-on-day changes in basis points:

```python
dM = np.diff(M, axis=0) * 100
dM.shape           # (4, 12): one fewer row
dM.round(1)
```

### Practice 2

1. Which tenor moved the most (largest absolute change) across the whole 5-day window, first day to last?
2. On which day did the curve as a whole move the most, measured by the average absolute change across tenors?
3. Compute the 2s10s slope (10Y minus 2Y, in bp) for each of the five days in one line.

<details><summary>Answer</summary>

```python
total = (M[-1] - M[0]) * 100
tenors[np.abs(total).argmax()], total[np.abs(total).argmax()]

day_move = np.abs(dM).mean(axis=1)
dates[1:][day_move.argmax()]           # dM's first row is the change INTO 4 August

(M[:, 9] - M[:, 5]) * 100
```
</details>

## 1.5 Broadcasting (20 min)

Broadcasting lets NumPy combine arrays of different shapes without copying data. The rule: compare shapes from the right; two dimensions are compatible if they are equal or one of them is 1.

```python
rates.shape, mats.shape          # (12,) and (12,): same shape, element by element
rates + 0.25                     # a scalar broadcasts to every element: a parallel shift
```

Now a real use: **scenario analysis**. Three shocks to the curve, in bp, one row per scenario:

```python
shocks = np.array([
    np.full(12, 25.0),                        # parallel +25bp
    np.linspace(-20, 20, 12),                 # steepener: short end down, long end up
    np.where(mats <= 2, 30.0, 0.0),           # front-end selloff: +30bp up to 2Y
])
shocks.shape                                  # (3, 12)

shocked = rates + shocks / 100                # (12,) + (3, 12) → (3, 12)
shocked.shape
```

`rates` has shape `(12,)`, which broadcasts as `(1, 12)` against `(3, 12)`: the same base curve is added to every scenario row.

The book's PV01 per tenor (from `positions.csv`, typed in here):

```python
pv01 = np.array([0, -500, 1200, -800, 2500, -9000, 4000, 15000, -6000, -12000, 3000, 5000])
scenario_pnl = shocks @ pv01          # matrix product: (3, 12) @ (12,) → (3,)
scenario_pnl
```

`@` is matrix multiplication (the same as `np.dot` here). Each scenario's P&L is the sum over tenors of shock × PV01.

A column vector broadcast against a row vector produces a grid:

```python
shift = np.array([-50, 0, 50])[:, None]     # shape (3, 1): None adds an axis
grid = rates + shift / 100                  # (12,) and (3, 1) → (3, 12)
grid.round(2)
```

`[:, None]` (or `.reshape(-1, 1)`) turns a 1-D array into a column. It's the single most useful broadcasting trick.

### Practice 3

1. Compute discount factors for all three shocked curves in one expression. The result should be `(3, 12)`.
2. A "bucketed" view: the P&L contribution of each tenor in each scenario, as a `(3, 12)` array. Which tenor drives the parallel scenario's P&L most?
3. Without a loop, subtract each tenor's 5-day mean from `M` (de-meaning each column).

<details><summary>Answer</summary>

```python
dfs = np.exp(-shocked / 100 * mats)
dfs.shape

contrib = shocks * pv01                 # (3, 12) * (12,) → (3, 12)
tenors[np.abs(contrib[0]).argmax()]     # 5Y: the largest PV01

M - M.mean(axis=0)                      # (5, 12) - (12,) broadcasts across rows
```
</details>

## 1.6 Missing values: NaN (10 min)

`np.nan` ("not a number") marks missing data. It spreads through arithmetic: anything combined with NaN is NaN.

```python
x = np.array([3.6, np.nan, 3.7, 3.8])
x.mean()                 # nan
np.nanmean(x)            # 3.7: ignores NaN
np.isnan(x)              # boolean mask of missing values
x[~np.isnan(x)]          # only the valid values
np.nan == np.nan         # False! never compare with ==, always use np.isnan
```

NaN is a float, so an integer array can't hold it. `np.array([1, 2, np.nan])` silently becomes floats.

## 1.7 Random numbers and simulation (15 min)

Always create a generator with a seed, so results are reproducible:

```python
rng = np.random.default_rng(42)
rng.normal(0, 1, 5)
rng.uniform(0, 1, 3)
rng.integers(0, 10, 4)
rng.choice(tenors, 3, replace=False)
```

(Older code uses `np.random.seed()` and `np.random.normal()`. `default_rng` is the modern, recommended interface.)

**Monte Carlo value-at-risk for the book.** Use the daily changes in `dM` to estimate how the tenors move together, simulate many days, and read off the loss at the 1st percentile:

```python
cov = np.cov(dM, rowvar=False)               # 12×12 covariance of daily bp changes
sim = rng.multivariate_normal(np.zeros(12), cov, size=100_000)   # 100,000 simulated days
sim_pnl = sim @ pv01
var_99 = -np.percentile(sim_pnl, 1)
round(var_99)
```

`rowvar=False` tells `np.cov` that each **column** is a variable. Forgetting it gives a 4×4 matrix of covariances between days, a classic bug.

Be honest about what this number is: a covariance estimated from **four** days of changes is almost meaningless (a 12×12 covariance needs far more observations than 4 to be reliable). Spotting that is exactly the kind of judgement an interviewer wants to hear. On Day 5 you'll redo it with all six weeks.

## 1.8 Linear algebra and regression (10 min)

```python
A = np.array([[2.0, 1.0], [1.0, 3.0]])
A @ A, A.T, np.linalg.inv(A), np.linalg.det(A)
np.linalg.solve(A, np.array([1.0, 2.0]))     # solves A x = b; better than inv(A) @ b
```

**Ordinary least squares with `lstsq`.** Does the 10Y move one-for-one with the 2Y? Regress the 10Y daily change on the 2Y daily change, using the four days in `dM`:

```python
x, y = dM[:, 5], dM[:, 9]                       # 2Y and 10Y changes
X = np.column_stack([np.ones_like(x), x])        # a column of 1s for the intercept
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
beta                                             # [intercept, slope]
np.corrcoef(x, y)[0, 1]
```

Four points is again too few to trust; Day 5 repeats this properly. What matters today is the mechanics: build a design matrix with an intercept column, then solve.

## 1.9 Views versus copies (5 min)

Slicing returns a **view**: a window onto the same memory, not a copy.

```python
a = np.arange(5)
v = a[1:3]
v[0] = 99
a                      # array([ 0, 99,  2,  3,  4]): the original changed!
c = a[1:3].copy()
c[0] = -1
a                      # unchanged this time
```

Boolean and fancy indexing (`a[mask]`, `a[[0, 2]]`) always return copies. When in doubt, call `.copy()`. pandas has its own version of this issue, covered on Day 2.

## 1.10 Mini-lab: discount factors and a no-arbitrage check (15 min)

A sound yield curve must have **discount factors that fall as maturity rises** (money later is worth less than money now) and, in normal markets, no wild swings in **forward rates**. The continuously compounded forward rate between maturities `t1` and `t2` is:

`f = (r2·t2 − r1·t1) / (t2 − t1)`

Using the 2 September and 3 September curves:

1. Load both curves with `np.genfromtxt`.
2. Compute discount factors for each. Are they strictly decreasing on both days? (`np.all(np.diff(...) < 0)`.)
3. Compute the forward rates between each pair of neighbouring tenors, in percent, for both days, using vectorised slicing (`[1:]` and `[:-1]`), not a loop.
4. Compare the two days' forwards. Which forward rates moved by more than 50bp overnight?
5. Is the 3 September curve arbitrage-free by the discount-factor test? Is it believable? What does that tell you about simple checks?

<details><summary>Answer</summary>

```python
c2 = np.genfromtxt("lab/data/curves/sofr_ois_2026-09-02.csv", delimiter=",", skip_header=1, usecols=2)
c3 = np.genfromtxt("lab/data/curves/sofr_ois_2026-09-03.csv", delimiter=",", skip_header=1, usecols=2)

df2, df3 = np.exp(-c2 / 100 * mats), np.exp(-c3 / 100 * mats)
np.all(np.diff(df2) < 0), np.all(np.diff(df3) < 0)     # True, True

def forwards(r, t):
    return (r[1:] * t[1:] - r[:-1] * t[:-1]) / (t[1:] - t[:-1])

f2, f3 = forwards(c2, mats), forwards(c3, mats)
labels = np.array([f"{a}-{b}" for a, b in zip(tenors[:-1], tenors[1:])])
moved = np.abs(f3 - f2) * 100
labels[moved > 50], moved[moved > 50].round(0)
```

Only one forward breaches 50bp: the 7Y–10Y forward jumps by about 148bp (and the 10Y–20Y forward drops by about 45bp; print `((f3 - f2) * 100).round(0)` to see every move). Discount factors are still decreasing, so the curve passes the basic no-arbitrage test, yet no real market reprices a forward by 150bp overnight while its neighbours barely move. The cause is a single bad 10Y quote. The lesson: a no-arbitrage check catches impossible curves, not implausible ones; you also need statistical checks on changes, which is Day 4.
</details>

## 1.11 Timed drill (10 min, no notes)

1. Create an array of the 12 maturities and the 3 September rates.
2. Print the tenors where the rate is below 3.75%.
3. Compute discount factors, then check they decrease.
4. Build the 5×12 matrix `M` again and print each tenor's range (max minus min) in bp.
5. A parallel +10bp shock: P&L of the book, in one expression.
6. Replace any value below 0.5 in an array with NaN, then take the mean ignoring NaN.

## Day 1 checklist

- [ ] Why is vectorised code faster than a Python loop?
- [ ] What does `axis=0` mean in `M.mean(axis=0)`?
- [ ] Why use `&` instead of `and` with arrays, and why the brackets?
- [ ] State the broadcasting rule. What does `[:, None]` do?
- [ ] Why does `np.nan == np.nan` return False, and what do you use instead?
- [ ] What does `rowvar=False` do in `np.cov`?
- [ ] How do you run an OLS regression in NumPy?
- [ ] When does slicing give a view, and why does it matter?

---
# Day 2: pandas Foundations

**Goal:** load many files into one DataFrame, inspect it quickly, get the types right, select and filter rows and columns without hesitation, add columns, handle missing values, and avoid the classic assignment trap. This is the day that makes everything else feel quick.
**Time:** about 2 hours.

## Warm-up (5 min, from memory, in `day2.ipynb`)

1. Create a NumPy array of the 3 September rates and the 12 maturities.
2. Print the tenors with rates above 4%.
3. Compute discount factors and check they decrease.
4. What does `axis=1` do in `M.sum(axis=1)`?

<details><summary>Answer</summary>

```python
import numpy as np
rates = np.genfromtxt("lab/data/curves/sofr_ois_2026-09-03.csv", delimiter=",", skip_header=1, usecols=2)
tenors = np.array(["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"])
mats = np.array([1/360, 1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
tenors[rates > 4]
np.all(np.diff(np.exp(-rates / 100 * mats)) < 0)
```

`axis=1` collapses the columns, giving one total per row.
</details>

Start the day's notebook with:

```python
import glob
import numpy as np
import pandas as pd
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
```

## 2.1 Series and DataFrame (10 min)

A **Series** is a 1-D array with labels (the **index**). A **DataFrame** is a table: a set of Series sharing one index, one per column.

```python
s = pd.Series([4.3753, 4.3186, 4.3001], index=["ON", "1M", "3M"], name="rate_pct")
s
s["1M"]                 # by label
s.iloc[1]               # by position
s * 100                 # vectorised, like NumPy, and the labels come along
s.to_numpy()            # back to a plain NumPy array
```

```python
df_small = pd.DataFrame({
    "tenor": ["ON", "1M", "3M"],
    "rate_pct": [4.3753, 4.3186, 4.3001],
})
df_small
df_small.columns, df_small.index
```

The key difference from NumPy: operations between pandas objects **align on labels**, not positions.

```python
a = pd.Series([1.0, 2.0], index=["ON", "1M"])
b = pd.Series([10.0, 20.0], index=["1M", "3M"])
a + b                   # ON: NaN, 1M: 12.0, 3M: NaN
```

Alignment is powerful (adding yesterday's and today's curves just works by tenor) but it produces NaN wherever labels don't match. When a calculation unexpectedly returns NaN, misaligned labels are the first thing to check.

## 2.2 Loading one file and inspecting it (15 min)

```python
one = pd.read_csv("lab/data/curves/sofr_ois_2026-09-03.csv")
one.head()             # first 5 rows; head(10) for more
one.tail(3)
one.shape              # (rows, columns)
one.dtypes
one.info()             # types, non-null counts, memory
one.describe()         # summary statistics of the numeric columns
```

Make `info()`, `describe()` and `head()` the first three things you run on any new dataset, and say so in an interview: "first I check shape, types, nulls and ranges."

**Types in pandas 3.** Text columns show as `str`. In pandas 2.x they show as `object`. Either way they hold strings. The `date` column is a string too, until you convert it.

## 2.3 Loading many files (15 min)

```python
files = sorted(glob.glob("lab/data/curves/*.csv"))
len(files), files[:2]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.shape
df.head()
```

- `glob` returns the matching paths; `sorted` makes the order predictable.
- `pd.concat` stacks DataFrames vertically. `ignore_index=True` gives a fresh 0…n−1 index instead of repeating each file's 0…11.
- **Never grow a DataFrame row by row in a loop** (`df = pd.concat([df, row])` inside a loop). Each step copies everything, so it gets slower and slower. Collect the pieces in a list, then concat once.

Keeping track of which file a row came from is useful when investigating:

```python
df_src = pd.concat(
    (pd.read_csv(f).assign(file=f.split("/")[-1]) for f in files),
    ignore_index=True,
)
df_src.head(3)
```

(On Windows, paths may use `\`. `pathlib.Path(f).name` works everywhere.)

## 2.4 Getting the types right (10 min)

```python
df["date"] = pd.to_datetime(df["date"])
df.dtypes
df["date"].min(), df["date"].max()
df["date"].dt.day_name().head(3)    # the .dt accessor works on datetime columns
```

Or parse dates while loading: `pd.read_csv(f, parse_dates=["date"])`.

**Tenor order.** Sorting tenors as text gives `10Y, 1M, 1Y, 20Y, ...`, which is useless for curves. An ordered categorical fixes that:

```python
tenor_order = ["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
df["tenor"] = pd.Categorical(df["tenor"], categories=tenor_order, ordered=True)
df.sort_values(["date", "tenor"]).head(12)["tenor"].tolist()
```

Categoricals also use far less memory for columns with few distinct values, which matters with millions of rows.

## 2.5 Selecting columns and rows (25 min)

**Columns:**

```python
df["rate_pct"]                       # one column → Series
df[["date", "tenor", "rate_pct"]]    # a list of columns → DataFrame
```

**Rows by position, `iloc`:**

```python
df.iloc[0]               # first row
df.iloc[:3]              # first three rows
df.iloc[:3, :2]          # first three rows, first two columns
df.iloc[-1, 2]           # last row, third column
```

**Rows by label and condition, `loc`:**

```python
df.loc[0, "rate_pct"]                         # index label 0, column rate_pct
df.loc[df["tenor"] == "10Y"]                  # boolean filter
df.loc[df["tenor"] == "10Y", ["date", "rate_pct"]]     # filter rows AND pick columns
df.loc[(df["rate_pct"] > 4.3) & (df["date"] >= "2026-09-01")]
```

The same rules as NumPy apply: `&`, `|`, `~`, with brackets around each condition.

**Handy filters:**

```python
df.loc[df["tenor"].isin(["2Y", "10Y"])]
df.loc[df["rate_pct"].between(3.70, 3.71)]
df.loc[df["source"] != "BBG"]
df.query("tenor == '10Y' and rate_pct > 4")     # SQL-like string; neat for interviews
```

**`loc` versus `iloc` in one sentence:** `loc` uses labels and conditions, and its slices *include* the end label; `iloc` uses integer positions, and its slices *exclude* the end, like Python.

### Practice 1

1. Show all rows for 12 August.
2. Show date and rate for every 30Y row above 4.1%.
3. How many rows are not from source BBG? Show them.
4. Show the 10 lowest rates in the whole dataset with date and tenor. (Hint: `nsmallest`.) Anything odd?

<details><summary>Answer</summary>

```python
df.loc[df["date"] == "2026-08-12"]
df.loc[(df["tenor"] == "30Y") & (df["rate_pct"] > 4.1), ["date", "rate_pct"]]
df.loc[df["source"] != "BBG"]           # one row, from BGN on 8 September
df.nsmallest(10, "rate_pct")[["date", "tenor", "rate_pct"]]
```

The lowest value, about 0.04% for the 1Y on 27 August, is around 100 times smaller than every other rate: almost certainly a unit error (a decimal stored instead of a percent).
</details>

## 2.6 Adding and changing columns (15 min)

```python
df["rate_bp"] = df["rate_pct"] * 100
df["rate_dec"] = df["rate_pct"] / 100
df["is_short_end"] = df["tenor"].isin(["ON", "1M", "3M", "6M"])
df["bucket"] = np.where(df["rate_pct"] > 4, "above 4%", "4% or below")
df.head(3)
```

`assign` does the same but returns a new DataFrame, which lets you chain steps:

```python
(df.assign(rate_bp=lambda d: d["rate_pct"] * 100)
   .loc[lambda d: d["tenor"] == "5Y", ["date", "rate_bp"]]
   .head(3))
```

**String and date accessors:**

```python
df["source"].str.lower().head(2)
df["snapshot_time"].str.startswith("16").all()        # False: something is off
df["date"].dt.isocalendar().week.head(2)
```

Drop columns you no longer need:

```python
df = df.drop(columns=["rate_bp", "rate_dec", "is_short_end", "bucket"])
df.columns.tolist()
```

## 2.7 The assignment trap: always use `.loc` (10 min)

This is one of the most common pandas interview questions. Suppose you want to replace implausible rates with NaN:

```python
test = df.copy()
test[test["rate_pct"] < 0.5]["rate_pct"] = np.nan      # chained assignment: WRONG
test["rate_pct"].isna().sum()
```

Under **pandas 3** this does nothing to `test` (and shows a `ChainedAssignmentError` warning): `test[...]` creates a new object, and you change that object, not `test`. In pandas 2.x it shows the famous `SettingWithCopyWarning`, and whether it works depends on internal details. Either way, it's a bug.

The correct form uses a single `.loc` call with rows and column together:

```python
test.loc[test["rate_pct"] < 0.5, "rate_pct"] = np.nan  # RIGHT
test["rate_pct"].isna().sum()                           # now 2: the original blank plus this one
```

Related: if you take a subset and plan to modify it, call `.copy()` to make your intention explicit:

```python
tens = df.loc[df["tenor"] == "10Y"].copy()
tens["rate_bp"] = tens["rate_pct"] * 100     # safe, changes only tens
```

pandas 3 uses **Copy-on-Write** throughout: any derived object behaves like a copy, and changes never leak back to the original. This makes behaviour predictable, but it's also why chained assignment silently fails to update the original.

## 2.8 Missing values (10 min)

```python
df.isna().sum()                       # missing values per column
df.loc[df["rate_pct"].isna()]         # the row with a blank rate
df["rate_pct"].notna().mean()         # share of non-missing values
df.dropna(subset=["rate_pct"]).shape  # drop rows missing a rate
df["rate_pct"].fillna(0).isna().sum() # fill with a value (fill with 0 is almost always wrong for prices!)
```

Filling a missing rate with 0 would create a fake 400bp move. For prices and rates, the honest choices are to leave it missing and flag it, interpolate with care (Day 4), or fall back to another source. Say this in an interview if a fill comes up.

## 2.9 Sorting and counting (10 min)

```python
df.sort_values("rate_pct", ascending=False).head(3)
df.sort_values(["tenor", "date"]).head(3)             # by tenor (categorical order), then date
df.nlargest(3, "rate_pct")

df["source"].value_counts()
df["snapshot_time"].value_counts()
df["tenor"].value_counts().sort_index()                # counts in tenor order
df["date"].nunique(), df["tenor"].unique()
```

`value_counts()` is the pandas version of `sort | uniq -c`. Run it on every text column when profiling a new dataset.

## 2.10 Saving (5 min)

```python
df.to_csv("curves_all.csv", index=False)     # index=False avoids an extra unnamed column
pd.read_csv("curves_all.csv").shape
```

For real pipelines, Parquet (`df.to_parquet("x.parquet")`, which needs `pip install pyarrow`) is faster, smaller and keeps the column types. CSV loses them: dates and categories come back as strings.

## 2.11 Mini-lab: profile the dataset (20 min)

Write a cell that loads all the files from scratch and prints a profile. It should answer, with code, not by eye:

1. How many rows, files, distinct dates and distinct tenors are there?
2. What is the date range? How many business days are listed in `lab/ref/business_days.txt`? Does the number of dates match?
3. How many rates are missing?
4. What sources and snapshot times appear, and how often?
5. What are the minimum and maximum rates? Is anything implausible (outside 0.5%–10%)?
6. Which dates do not have exactly 12 rows? (Hint: `df["date"].value_counts()`.)

Write down every problem you find. You should find at least five different kinds; Day 3 tracks them all down systematically.

<details><summary>Answer</summary>

```python
files = sorted(glob.glob("lab/data/curves/*.csv"))
df = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)
bdays = pd.read_csv("lab/ref/business_days.txt", header=None, names=["date"], parse_dates=["date"])

print("rows", len(df), "| files", len(files), "| dates", df["date"].nunique(), "| tenors", df["tenor"].nunique())
print("range", df["date"].min().date(), "to", df["date"].max().date(), "| expected business days", len(bdays))
print("missing rates", df["rate_pct"].isna().sum())
print(df["source"].value_counts().to_dict(), df["snapshot_time"].value_counts().to_dict())
print("min", df["rate_pct"].min(), "max", df["rate_pct"].max())
print(df.loc[~df["rate_pct"].between(0.5, 10) & df["rate_pct"].notna()])
counts = df["date"].value_counts()
print(counts[counts != 12])
```

Findings: 29 files but 30 business days (one day missing); one blank rate; one non-BBG row; one snapshot at 21:00; one rate around 0.04%; 12 August has 11 rows and 8 September has 13.

Note the `notna()` in the implausible-rate filter: `between` returns False for NaN, so without it the blank row would also appear as "outside the range".
</details>

## 2.12 Timed drill (10 min, no notes)

1. Load all curve files with dates parsed, into `df`.
2. Make `tenor` an ordered categorical.
3. Show the 2Y history (date and rate) sorted by date.
4. Add a `rate_bp` column, then show rows where it's above 435.
5. Replace rates outside 0.5%–10% with NaN using `.loc`.
6. Count rows per source.

## Day 2 checklist

- [ ] What is the index, and what does alignment mean?
- [ ] What three methods do you run first on new data?
- [ ] Why collect DataFrames in a list and concat once?
- [ ] Why make `tenor` an ordered categorical?
- [ ] `loc` versus `iloc`: what's the difference, including slice ends?
- [ ] Why is `df[mask]["col"] = x` wrong, and what's the right form?
- [ ] What is Copy-on-Write in pandas 3?
- [ ] Why is `fillna(0)` dangerous for market data?

---
# Day 3: groupby, Reshape and Merge

**Goal:** summarise by group, reshape between long and wide formats, join datasets, and use these tools to find every structural problem in the data: missing days, missing tenors, duplicates and source conflicts. By the end you will produce a tidy table of issues, the kind of output a monitoring system feeds to people.
**Time:** about 2 hours.

## Warm-up (5 min, from memory, in `day3.ipynb`)

1. Load all curve files with dates parsed.
2. Show the 5 highest rates with date and tenor.
3. Count rows per snapshot time.
4. Set rates below 0.5 to NaN using a single `.loc` call.

<details><summary>Answer</summary>

```python
import glob
import numpy as np
import pandas as pd
pd.set_option("display.width", 120)

files = sorted(glob.glob("lab/data/curves/*.csv"))
df = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)
df.nlargest(5, "rate_pct")[["date", "tenor", "rate_pct"]]
df["snapshot_time"].value_counts()
tmp = df.copy()
tmp.loc[tmp["rate_pct"] < 0.5, "rate_pct"] = np.nan
```
</details>

Set up for the day (run this even if you did the warm-up):

```python
tenor_order = ["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
df["tenor"] = pd.Categorical(df["tenor"], categories=tenor_order, ordered=True)
bdays = pd.read_csv("lab/ref/business_days.txt", header=None, names=["date"], parse_dates=["date"])
pos = pd.read_csv("lab/data/positions.csv")
```

## 3.1 groupby: split, apply, combine (25 min)

`groupby` splits the rows into groups, applies a function to each group, and combines the results. It's SQL's `GROUP BY`.

```python
df.groupby("tenor", observed=True)["rate_pct"].mean()
df.groupby("date")["rate_pct"].count()            # count ignores NaN
df.groupby("date").size()                         # size counts rows, including NaN
```

`count` versus `size` is a classic interview question, and it matters here: on 18 August, `size` is 12 but `count` is 11 because of the blank rate.

`observed=True` tells pandas to show only categories that actually appear. With a categorical key it's good practice to pass it explicitly.

**Several statistics at once with `agg`:**

```python
df.groupby("tenor", observed=True)["rate_pct"].agg(["mean", "std", "min", "max", "count"])
```

**Named aggregations**, which give clean column names and are the most readable form:

```python
summary = df.groupby("tenor", observed=True).agg(
    avg_rate=("rate_pct", "mean"),
    lo=("rate_pct", "min"),
    hi=("rate_pct", "max"),
    n=("rate_pct", "size"),
)
summary
```

Look at the 1Y row: its minimum is around 0.04%. One bad value distorts the mean of the whole group, and the standard deviation even more. That's why robust statistics (medians) matter in monitoring, which Day 4 uses.

**Grouping by more than one key:**

```python
df.groupby(["date", "source"]).size().head()
```

**`transform` returns one value per original row**, aligned to the input, rather than one per group. Use it to compare each row with its group:

```python
df["tenor_median"] = df.groupby("tenor", observed=True)["rate_pct"].transform("median")
df["dev_bp"] = (df["rate_pct"] - df["tenor_median"]) * 100
df.loc[df["dev_bp"].abs() > 30, ["date", "tenor", "rate_pct", "dev_bp"]]
```

This "deviation from the group median" check finds the unit error immediately and also flags the bad 10Y print. But it compares each rate with its *six-week* median, so it would also flag a genuine large market move that persisted. Comparing with recent history instead (rolling windows) is Day 4.

```python
df = df.drop(columns=["tenor_median", "dev_bp"])
```

**`filter` keeps or drops whole groups:**

```python
df.groupby("date").filter(lambda g: len(g) != 12)[["date", "tenor", "rate_pct", "source"]]
```

### Practice 1

1. For each date, compute the curve's average rate and the number of non-missing rates.
2. For each tenor, compute the range (max minus min) in bp. Which two tenors have suspiciously large ranges?
3. Using `transform`, add a column with each date's maximum rate, then show the rows where the rate equals it. Which tenor is usually the highest?

<details><summary>Answer</summary>

```python
df.groupby("date").agg(avg=("rate_pct", "mean"), n=("rate_pct", "count"))

rng = df.groupby("tenor", observed=True)["rate_pct"].agg(lambda s: (s.max() - s.min()) * 100)
rng.sort_values(ascending=False).head(3)          # 1Y (unit error) and 10Y (bad print)

df["day_max"] = df.groupby("date")["rate_pct"].transform("max")
df.loc[df["rate_pct"] == df["day_max"], "tenor"].value_counts().head(3)   # ON, nearly every day
df = df.drop(columns="day_max")
```
</details>

## 3.2 Long and wide: pivot, melt, stack (25 min)

Our data is in **long** format: one row per date and tenor. Many analyses want **wide** format: one row per date, one column per tenor. Moving between the two is essential.

```python
wide = df.pivot(index="date", columns="tenor", values="rate_pct")
```

That fails with `ValueError: Index contains duplicate entries`. `pivot` requires exactly one value per (date, tenor) cell, and the duplicate 5Y on 8 September breaks it. That error is itself a data-quality check. Find the culprit:

```python
dups = df[df.duplicated(["date", "tenor"], keep=False)]
dups
```

`keep=False` marks *every* copy of a duplicate, not just the second, so you see both rows.

Two ways forward. Either decide which row wins (for example, prefer Bloomberg) and drop the other, or aggregate with `pivot_table`:

```python
wide = (df[df["source"] == "BBG"]
          .pivot(index="date", columns="tenor", values="rate_pct"))
wide.shape
wide.head(3)
```

```python
pivot_mean = df.pivot_table(index="date", columns="tenor", values="rate_pct",
                            aggfunc="mean", observed=True)
pivot_mean.loc["2026-09-08", "5Y"]     # the average of BBG and BGN: silently blends two sources
```

`pivot_table` never fails on duplicates; it aggregates them. That's convenient and dangerous: it hid a 7bp disagreement between vendors without telling you. In an interview, say you'd check for duplicates before pivoting rather than let an aggregation absorb them.

Wide format makes curve calculations natural:

```python
wide["2s10s_bp"] = (wide["10Y"] - wide["2Y"]) * 100
wide["2s10s_bp"].describe()
wide = wide.drop(columns="2s10s_bp")
```

**Back to long with `melt` or `stack`:**

```python
long_again = wide.reset_index().melt(id_vars="date", var_name="tenor", value_name="rate_pct")
long_again.shape                     # 29 dates × 12 tenors = 348 rows
long_again["rate_pct"].isna().sum()  # 2: the blank 3Y and the missing 7Y now both show as NaN
```

Notice the missing 7Y on 12 August. In the long data it was simply an *absent row*; after pivoting and melting it became an explicit NaN. **Reshaping to a complete grid is a standard way to make missing data visible.**

`stack` does the same from a wide frame, but drops NaN in older pandas versions; `wide.stack()` gives a Series with a (date, tenor) MultiIndex.

### Practice 2

1. Build the wide frame, then show only 10Y, 20Y and 30Y for the first week of September.
2. In the wide frame, find every (date, tenor) cell that is NaN. (Hint: `wide.isna().stack()`.)
3. Using `pivot_table` with `aggfunc="count"`, build a date × tenor grid of row counts. Find every cell that isn't 1.

<details><summary>Answer</summary>

```python
wide.loc["2026-09-01":"2026-09-04", ["10Y", "20Y", "30Y"]]

na = wide.isna().stack()
na[na]                              # 2026-08-12 7Y and 2026-08-18 3Y

cnt = df.pivot_table(index="date", columns="tenor", values="source",
                     aggfunc="count", observed=False, fill_value=0)
cnt.stack()[lambda s: s != 1]        # 7Y on 12 Aug: 0; 5Y on 8 Sep: 2
```

Counting `source` rather than `rate_pct` matters: `count` ignores NaN, so counting the rate would wrongly report the blank 3Y as missing too.
</details>

## 3.3 Merging and joining (25 min)

`merge` is SQL's `JOIN`. Join the book's PV01 onto every curve row:

```python
pos.head(3)
m = df.merge(pos, on="tenor", how="left")
m.head(3)
```

(`df["tenor"]` is categorical and `pos["tenor"]` is a string. pandas handles it, but matching types on join keys avoids surprises.)

The `how` argument:

| `how` | Keeps |
|---|---|
| `inner` | only keys in both |
| `left` | every row of the left table |
| `right` | every row of the right table |
| `outer` | every key from either side |

**Finding gaps with `indicator=True`.** Which business days have no data at all?

```python
dates_present = df[["date"]].drop_duplicates()
check = bdays.merge(dates_present, on="date", how="left", indicator=True)
check["_merge"].value_counts()
check.loc[check["_merge"] == "left_only", "date"]
```

`_merge` tells you which side each row came from. `left_only` here means "a business day with no curve", which is exactly the missing file.

**Always check row counts before and after a merge.** A merge on a key that isn't unique on one side multiplies rows silently:

```python
dup_pos = pd.concat([pos, pos.iloc[[7]]])          # 5Y listed twice by mistake
len(df), len(df.merge(dup_pos, on="tenor"))        # rows grew: every 5Y row doubled
```

`validate` makes pandas raise an error instead:

```python
df.merge(pos, on="tenor", validate="many_to_one").shape     # passes: pos has one row per tenor
```

Try `validate="many_to_one"` with `dup_pos`, and read the error. Put `validate=` on merges in production code; it turns a silent data bug into a loud one.

**Building a complete grid.** Every business day crossed with every tenor, the "expected" dataset:

```python
grid = pd.MultiIndex.from_product([bdays["date"], tenor_order], names=["date", "tenor"]).to_frame(index=False)
len(grid)                                                       # 30 × 12 = 360
full = grid.merge(df.astype({"tenor": str}), on=["date", "tenor"], how="left", indicator=True)
full.loc[full["_merge"] == "left_only", ["date", "tenor"]]
```

That one query finds the missing 7Y *and* all twelve tenors of the missing day.

`pd.concat` stacks frames; `merge` lines them up side by side by key. `df.join(other)` is a shortcut for merging on the index.

### Practice 3

1. Join PV01 onto the data, then compute for each date the sum of `pv01_usd × rate_pct`. (It's meaningless as a number; it's practice for tomorrow's P&L.)
2. For 8 September, put the BBG and BGN 5Y quotes side by side in one row and compute the difference in bp. (Hint: filter, then `pivot` on `source`.)
3. Using an `outer` merge with `indicator=True`, compare the tenors in `pos` with the tenors present on 12 August.

<details><summary>Answer</summary>

```python
m = df.merge(pos, on="tenor", how="left", validate="many_to_one")
(m["pv01_usd"] * m["rate_pct"]).groupby(m["date"]).sum().head(3)

conflict = (df[(df["date"] == "2026-09-08") & (df["tenor"] == "5Y")]
              .pivot(index=["date", "tenor"], columns="source", values="rate_pct"))
conflict["diff_bp"] = (conflict["BGN"] - conflict["BBG"]) * 100
conflict                                         # 7.0bp

aug12 = df.loc[df["date"] == "2026-08-12", ["tenor"]].astype(str)
pos[["tenor"]].merge(aug12, on="tenor", how="outer", indicator=True).query("_merge != 'both'")
```
</details>

## 3.4 `apply`: useful, but last resort (10 min)

`apply` runs a Python function on each row, column or group. It's flexible but slow, because it's a loop in disguise.

```python
%timeit df["rate_pct"].apply(lambda x: x * 100)
%timeit df["rate_pct"] * 100
```

Even on 348 rows the vectorised version is much faster; on millions of rows the gap is the difference between seconds and minutes. Never use `iterrows` for calculations. Reach for `apply` only when no vectorised method exists, such as a custom calculation on each group:

```python
def curve_shape(g):
    r = g.set_index("tenor")["rate_pct"]
    return pd.Series({"slope_2s10s": (r.get("10Y") - r.get("2Y")) * 100,
                      "inverted_front": r.get("2Y") < r.get("ON")})

df[df["source"] == "BBG"].groupby("date")[["tenor", "rate_pct"]].apply(curve_shape).head()
```

Even this is cleaner as a pivot followed by column arithmetic, which is how you'd write it in production.

## 3.5 Mini-lab: a tidy table of structural issues (25 min)

Build a single DataFrame called `issues` with columns `date`, `tenor`, `check` and `detail`, one row per problem. Use the tools from today, and no manual inspection. It must include:

1. **missing_day**: business days with no data at all (one row per day, `tenor` left empty).
2. **missing_tenor**: expected (date, tenor) pairs with no row, on days that do have data.
3. **blank_value**: rows with a missing rate.
4. **duplicate**: (date, tenor) pairs with more than one row, with the sources in `detail`.
5. **source_conflict**: for duplicates, the disagreement in bp.
6. **off_snapshot**: snapshot time other than 16:00.
7. **implausible**: rates outside 0.5%–10%.

Build each check as a small DataFrame with the four columns, then `pd.concat` them and sort by date. Aim for one or two lines of pandas per check.

<details><summary>Answer</summary>

```python
cols = ["date", "tenor", "check", "detail"]
df_s = df.astype({"tenor": str})

present = df_s[["date"]].drop_duplicates()
md = bdays.merge(present, how="left", indicator=True).query("_merge == 'left_only'")
missing_day = md.assign(tenor="", check="missing_day", detail="no file")[cols]

grid = pd.MultiIndex.from_product([present["date"], tenor_order], names=["date", "tenor"]).to_frame(index=False)
mt = grid.merge(df_s[["date", "tenor"]].drop_duplicates(), how="left", indicator=True).query("_merge == 'left_only'")
missing_tenor = mt.assign(check="missing_tenor", detail="row absent")[cols]

blank = df_s[df_s["rate_pct"].isna()].assign(check="blank_value", detail="rate is empty")[cols]

d = df_s[df_s.duplicated(["date", "tenor"], keep=False)]
dup = (d.groupby(["date", "tenor"])
         .agg(detail=("source", lambda s: "sources: " + ", ".join(s)),
              spread=("rate_pct", lambda s: (s.max() - s.min()) * 100))
         .reset_index())
duplicate = dup.assign(check="duplicate")[cols]
conflict = dup.assign(check="source_conflict", detail=dup["spread"].map("{:.1f}bp".format))[cols]

off = df_s[df_s["snapshot_time"] != "16:00"]
off_snapshot = off.assign(check="off_snapshot", detail="snapshot " + off["snapshot_time"])[cols]

bad = df_s[df_s["rate_pct"].notna() & ~df_s["rate_pct"].between(0.5, 10)]
implausible = bad.assign(check="implausible", detail="rate " + bad["rate_pct"].astype(str))[cols]

issues = (pd.concat([missing_day, missing_tenor, blank, duplicate, conflict, off_snapshot, implausible])
            .sort_values(["date", "check"]).reset_index(drop=True))
issues
```

You should get 7 rows, one per check: missing day 21 August, missing 7Y on 12 August, blank 3Y on 18 August, the unit error on 27 August, the 5Y duplicate and its 7.0bp conflict on 8 September, and the 21:00 snapshot on 10 September.

Two problems from the bash lab are not here: the **stale** 20Y and the **bad 10Y print**. Both are plausible values in isolation; you can only see them by comparing a value with its own history. That's tomorrow.
</details>

## 3.6 Timed drill (10 min, no notes)

1. Average and standard deviation of rate per tenor, with named aggregations.
2. Dates whose row count isn't 12, using `groupby` and `size`.
3. Wide frame (BBG only), then the 5s30s slope in bp per date.
4. Find business days with no data using `merge(..., indicator=True)`.
5. Show both rows of every duplicate (date, tenor).
6. Explain to an imaginary interviewer, in two sentences, why `pivot_table` can be dangerous.

## Day 3 checklist

- [ ] What's the difference between `count` and `size` in a groupby?
- [ ] What's the difference between `agg` and `transform`?
- [ ] Why did `pivot` fail, and why is that useful?
- [ ] When is `pivot_table` dangerous?
- [ ] What does `keep=False` do in `duplicated`?
- [ ] How do you find keys present in one table but not another?
- [ ] How can a merge silently multiply rows, and how does `validate=` help?
- [ ] Why is `apply` slow, and when is it acceptable?

---
# Day 4: Time Series and Anomaly Detection

**Goal:** work with dates as an index, compare each value with its own history (`diff`, `shift`, `rolling`), build robust anomaly scores, detect stale data, resample, align timestamps with `merge_asof`, and plot. By the end you will have a data-health monitor that finds the problems Day 3 couldn't.
**Time:** about 2 hours. This is the most important day for the interview.

## Warm-up (5 min, from memory, in `day4.ipynb`)

1. Load all curve files with dates parsed.
2. Build a wide frame (date × tenor) from the BBG rows, with tenors in curve order.
3. Find all NaN cells in it.
4. Find business days with no data.

<details><summary>Answer</summary>

```python
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 120)

tenor_order = ["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
df = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob("lab/data/curves/*.csv"))),
               ignore_index=True)
wide = df[df["source"] == "BBG"].pivot(index="date", columns="tenor", values="rate_pct")[tenor_order]
wide.isna().stack()[lambda s: s]

bdays = pd.to_datetime(pd.read_csv("lab/ref/business_days.txt", header=None)[0]).rename("date")
bdays[~bdays.isin(wide.index)]
```

Selecting `[tenor_order]` after the pivot puts the columns in curve order.
</details>

## 4.1 A proper time index (15 min)

The wide frame's index is already dates, but one business day is missing entirely, so the rows aren't evenly spaced. Reindex onto the full calendar:

```python
wide.index.dtype, len(wide)
wide = wide.reindex(pd.DatetimeIndex(bdays))
wide.index.name = "date"
len(wide)
wide.loc["2026-08-20":"2026-08-24", ["2Y", "10Y"]]
```

Now 21 August is an explicit row of NaN. This matters: without it, a day-on-day change on 24 August would silently compare with 20 August, a *two*-day move, and mix it in with one-day moves.

**Selecting by date** with a DatetimeIndex is flexible:

```python
wide.loc["2026-09-03"]                          # one day
wide.loc["2026-09"]                             # the whole of September (partial-string indexing)
wide.loc["2026-08-25":"2026-08-28", "20Y"]      # a range; both ends included
wide.index.dayofweek[:5], wide.index.month[:3]
```

**Calendars.** `pd.bdate_range("2026-08-03", "2026-09-11")` gives Monday–Friday dates. It knows nothing about holidays: 7 September 2026 is US Labor Day, when SOFR isn't published, yet it's in `bdate_range` and also in our feed. Real systems use a proper holiday calendar for each market. Mentioning this in an interview shows you've worked with real data.

## 4.2 `diff`, `shift` and changes (15 min)

```python
chg = wide.diff() * 100            # day-on-day change in bp
chg.loc["2026-09-01":"2026-09-04", ["2Y", "10Y", "20Y"]].round(1)
```

`diff()` is `x - x.shift(1)`. `shift` moves the data down by one row, so each row can see the previous one:

```python
pd.DataFrame({"today": wide["10Y"], "yesterday": wide["10Y"].shift(1)}).loc["2026-09-02":"2026-09-04"]
```

`shift(-1)` looks one row *ahead*. Using future values in a signal or a model is **look-ahead bias**, one of the most serious mistakes in quantitative research, and systematic funds probe for it. When you write `shift(-1)`, stop and ask whether that information would really have been available at the time.

**Basis-point changes, not percentage changes.** `pct_change()` on rates gives the relative change in the rate itself (4.00% to 4.04% is a 1% change), which is rarely what anyone means. Rates are compared in bp; prices and index levels in percent or log returns.

Find big moves:

```python
moves = chg.stack()
moves[moves.abs() > 15].round(1)
```

Four flags: a 1Y fall and rebound around the unit error, and a 10Y jump and reversal around the bad print. A single bad point produces **two** flags, because the next day's change is measured from the bad value. Note too that a fixed 15bp threshold knows nothing about how volatile each tenor normally is.

## 4.3 Rolling windows (20 min)

```python
r10 = wide["10Y"]
pd.DataFrame({
    "rate": r10,
    "ma5": r10.rolling(5).mean(),
    "ma5_min3": r10.rolling(5, min_periods=3).mean(),
}).head(7)
```

- `rolling(5)` uses the current row and the 4 before it. It's **trailing** by default, so no look-ahead.
- The first 4 rows are NaN because the window isn't full. `min_periods` relaxes that.
- `rolling(5, center=True)` uses 2 rows on each side. That's fine for plotting history, but it's look-ahead in a live monitor or a backtest.

Rolling statistics on all columns at once:

```python
vol = chg.rolling(10, min_periods=5).std()
vol.iloc[-1].round(2)              # recent daily volatility per tenor, in bp
```

Look at the 10Y: its "volatility" is about 21bp, ten times its neighbours, because the bad print and its reversal still sit inside the window. (The 1Y looks normal again by now; its outliers have dropped out.) A standard deviation is not robust: one bad point distorts it until it drops out of the window.

Other windows: `rolling("7D")` uses a calendar-time window, which handles irregular spacing; `ewm(span=10).mean()` is an exponentially weighted average that reacts faster and has no hard cut-off; `expanding()` uses everything up to each row.

## 4.4 Anomaly scores: z-score versus robust z-score (25 min)

A **z-score** measures how unusual today's change is compared with recent history:

`z = (change − rolling mean) / rolling std`

```python
mu = chg.rolling(10, min_periods=5).mean().shift(1)
sd = chg.rolling(10, min_periods=5).std().shift(1)
z = (chg - mu) / sd
z.stack()[lambda s: s.abs() > 4].round(1)
```

The `.shift(1)` matters: it builds the statistics from **previous** days only, so today's value doesn't dilute its own score. Without it, a large outlier inflates the very standard deviation it's measured against and partly hides itself.

The ordinary z-score has a deeper weakness. After the bad 10Y print, the window's standard deviation is huge for the next ten days, so genuine problems in that period become invisible. The **robust z-score** uses the median and the median absolute deviation (MAD), which a few outliers barely affect:

`robust z = (change − rolling median) / (1.4826 × rolling MAD)`

The factor 1.4826 makes the MAD comparable with a standard deviation for normally distributed data.

```python
med = chg.rolling(10, min_periods=5).median().shift(1)
mad = (chg - med).abs().rolling(10, min_periods=5).median().shift(1)
rz = (chg - med) / (1.4826 * mad)
rz.stack()[lambda s: s.abs() > 6].round(1)
```

Look at the 20Y entries. On 2 September the 20Y moved only about 4bp, yet it scores around −13. During the stale run most of its recent changes were exactly zero, so its MAD collapsed towards zero and an ordinary move looks extreme. Two lessons:

1. **Floor the denominator** so near-zero dispersion can't produce huge (or, if the MAD is exactly zero, infinite) scores:

```python
mad_floor = mad.clip(lower=0.5)     # never assume less than 0.5bp of typical daily movement
rz = (chg - med) / (1.4826 * mad_floor)
flags = rz.stack()[lambda s: s.abs() > 6].round(1)
flags
```

   With the floor, only the 1Y and 10Y problems remain. The 20Y move drops below the threshold.

2. **The 20Y move was a symptom, not the problem.** When the stale quote finally updated, a week of accumulated movement arrived in one day. The real problem is the week of identical values *before* it, and a jump detector is the wrong tool for that. Stale data needs its own check, which is section 4.5. Different failure modes need different detectors.

The choice of window, threshold and floor are judgements, not facts. Expect an interviewer to ask how you would choose them: by backtesting the monitor on history with known incidents, measuring false positives versus missed problems, and setting different thresholds per tenor or per regime.

### Practice 1

1. Recompute the robust z-score with a 20-day window. Which flags disappear or appear? Why?
2. Count flags per tenor for thresholds of 4, 6 and 8. How sensitive is the monitor to the threshold?
3. Explain in two sentences why `shift(1)` belongs before the rolling statistics in a live monitor.

<details><summary>Answer</summary>

```python
def robust_z(chg, window=10, floor=0.5):
    med = chg.rolling(window, min_periods=5).median().shift(1)
    mad = (chg - med).abs().rolling(window, min_periods=5).median().shift(1)
    return (chg - med) / (1.4826 * mad.clip(lower=floor))

robust_z(chg, 20).stack()[lambda s: s.abs() > 6].round(1)

for thr in (4, 6, 8):
    print(thr, (robust_z(chg).abs() > thr).sum()[lambda s: s > 0].to_dict())
```

With only six weeks of data, a 20-day window leaves most of the history with too few observations, so behaviour changes mainly at the start. With the shift, today's statistics use only information available before today's value arrived, which is how a live monitor works and why scores aren't diluted by the value being tested.
</details>

## 4.5 Stale data (15 min)

A stale quote is a value that stops changing. Real rates move almost every day, at least by a fraction of a basis point.

```python
unchanged = wide.diff().eq(0)
unchanged.sum()[lambda s: s > 0]                       # days with zero change, per tenor
stale = unchanged.rolling(4).sum().eq(4)               # 4 zero changes in a row = 5 identical values
stale.stack()[lambda s: s]
```

The 20Y is flagged from 28 August. But the value actually stopped changing on 24 August. Like the bash lab, the window introduces a **detection lag**: you need five identical values before you're sure.

Find the true start of each stale run by walking back from the flag:

```python
run_id = (~unchanged["20Y"]).cumsum()                  # a new id every time the value changes
run_len = unchanged["20Y"].groupby(run_id).cumsum()    # length of the current unchanged run
pd.DataFrame({"20Y": wide["20Y"], "unchanged": unchanged["20Y"], "run_len": run_len}).loc["2026-08-21":"2026-09-02"]
```

The run starts on 24 August (the last real value) and the first stale value is 25 August. The `cumsum` trick, which creates a group id that increments whenever a condition breaks, is a classic pattern for finding consecutive runs; learn it.

Backfill the flag over the whole run:

```python
stale_days = unchanged["20Y"] & run_id.isin(run_id[stale["20Y"]])
stale_days[stale_days].index.date
```

`stale_days` marks every day inside a run that eventually became long enough to flag: 25 August to 1 September. In a live system you only learn this on 28 August, but for cleaning history, and for explaining P&L tomorrow, you want the whole run.

**A trade-off worth stating in an interview:** a shorter window catches stale data sooner but flags quiet, illiquid tenors that genuinely didn't move. A common fix is to compare with related tenors: if 10Y and 30Y moved 3bp and 20Y moved exactly zero, the 20Y is suspicious even after one day.

## 4.6 Resampling (10 min)

`resample` changes the frequency: daily to weekly, for example.

```python
weekly_close = wide.resample("W-FRI").last()       # each week's last available value
weekly_close[["2Y", "10Y"]].round(3)
weekly_avg = wide.resample("W-FRI").mean()
weekly_range = (wide.resample("W-FRI").max() - wide.resample("W-FRI").min()) * 100
weekly_range.round(1).loc[:, ["1Y", "10Y", "20Y"]]
```

`W-FRI` means weeks ending on Friday. Note that `last()` takes the last *non-missing* value in each week, which quietly hides a missing Friday. And the weekly range picks up the bad 1Y and 10Y values immediately, a quick sanity view for a report.

## 4.7 Timestamps, time zones and `merge_asof` (15 min)

Our data has one snapshot per day, but much market data is intraday. `merge_asof` joins each row to the **most recent** row of another table at or before its timestamp. It's how you attach the prevailing quote to each trade.

```python
quotes = pd.DataFrame({
    "time": pd.to_datetime(["2026-09-03 15:59:58", "2026-09-03 16:00:00",
                            "2026-09-03 16:00:03", "2026-09-03 16:00:09"]),
    "quote_10Y": [3.851, 3.853, 3.850, 3.858],
})
trades = pd.DataFrame({
    "time": pd.to_datetime(["2026-09-03 15:59:59", "2026-09-03 16:00:05", "2026-09-03 16:00:30"]),
    "trade_rate": [3.852, 3.851, 3.860],
})
pd.merge_asof(trades, quotes, on="time")                                  # last quote at or before
pd.merge_asof(trades, quotes, on="time", tolerance=pd.Timedelta("5s"))    # too old → NaN
pd.merge_asof(trades, quotes, on="time", direction="forward")             # next quote: look-ahead!
```

Both tables must be sorted by the key. Use `by="symbol"` to match within each instrument. The `tolerance` version is the important one in practice: a quote from 21 seconds ago is stale for a trade, and NaN is more honest than an old price.

**Time zones.** The snapshot is "16:00 London", but a New York system logs times in Eastern time:

```python
t = pd.Timestamp("2026-09-03 16:00").tz_localize("Europe/London")
t, t.tz_convert("America/New_York"), t.tz_convert("UTC")
```

London and New York change clocks on different dates in spring and autumn, so the gap between them is 5 hours most of the year but 4 hours for a few weeks. Store timestamps in UTC and convert only for display. That's also the root of problems like the "21:00 snapshot" in our data: 21:00 London is 16:00 New York, a plausible cause of that mismatch.

## 4.8 Plotting (10 min)

pandas plots directly with matplotlib:

```python
ax = wide[["2Y", "10Y", "30Y"]].plot(figsize=(9, 4), title="SOFR OIS rates (raw feed)")
ax.set_ylabel("rate, %")
plt.tight_layout()
plt.savefig("rates_raw.png", dpi=120)
```

The 10Y spike on 3 September is obvious by eye. Plotting first is good practice; then turn what you saw into a check you can automate.

A heatmap-style view of the robust scores:

```python
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(rz.abs().clip(upper=10).T, aspect="auto", cmap="Reds")
ax.set_yticks(range(len(tenor_order)), tenor_order)
ax.set_xticks(range(0, len(rz), 5), rz.index[::5].strftime("%d %b"))
fig.colorbar(im, label="|robust z|, capped at 10")
ax.set_title("Data-health scores by tenor and day")
plt.tight_layout()
plt.savefig("health_heatmap.png", dpi=120)
```

A chart like this is a strong thing to show in a report to non-technical leadership: problems appear as dark cells.

## 4.9 Mini-lab: a data-health monitor (25 min)

Write a function:

```python
def health_check(wide, z_window=10, z_threshold=6, mad_floor=0.5, stale_len=5):
    """Return a DataFrame of issues with columns date, tenor, check, detail."""
    ...
```

It takes the wide, reindexed frame and returns one row per issue for:

1. **jump**: robust z-score above the threshold (shifted rolling median and MAD, with a floor). Put the change in bp and the score in `detail`.
2. **stale**: a value unchanged for `stale_len` consecutive observations. Report every day in the run, not just from the detection date.
3. **missing**: NaN cells (this catches the missing day, the missing tenor and the blank value at once, because the frame is reindexed).

Then:

4. Run it on `wide` and combine its output with yesterday's `issues` table (rebuild it, or save it from Day 3 with `issues.to_csv("issues_day3.csv", index=False)`).
5. Which of the eight problems from the dataset does each check catch? Is anything caught twice? Is anything still missed?
6. Which flags are **consequences** of an earlier problem rather than new problems (for example, the day after a bad print)? How would you suppress them?

<details><summary>Answer</summary>

```python
def health_check(wide, z_window=10, z_threshold=6, mad_floor=0.5, stale_len=5):
    chg = wide.diff() * 100
    med = chg.rolling(z_window, min_periods=5).median().shift(1)
    mad = (chg - med).abs().rolling(z_window, min_periods=5).median().shift(1)
    rz = (chg - med) / (1.4826 * mad.clip(lower=mad_floor))

    out = []

    j = rz.stack()
    j = j[j.abs() > z_threshold]
    for (d, t), score in j.items():
        out.append((d, t, "jump", f"{chg.loc[d, t]:+.1f}bp, robust z {score:+.1f}"))

    unchanged = wide.diff().eq(0)
    for t in wide.columns:
        u = unchanged[t]
        run_id = (~u).cumsum()
        run_len = u.groupby(run_id).cumsum()
        long_runs = run_id[run_len >= stale_len - 1].unique()
        for d in u.index[u & run_id.isin(long_runs)]:
            out.append((d, t, "stale", f"unchanged at {wide.loc[d, t]}"))

    m = wide.isna().stack()
    for (d, t) in m[m].index:
        out.append((d, t, "missing", "no value"))

    return (pd.DataFrame(out, columns=["date", "tenor", "check", "detail"])
              .sort_values(["date", "tenor", "check"]).reset_index(drop=True))

h = health_check(wide)
h["check"].value_counts()
h
```

Findings: **missing** catches the missing day (12 cells), the missing 7Y and the blank 3Y. **stale** catches the 20Y from 25 August to 1 September. **jump** catches the unit error and its rebound, and the bad 10Y print and its reversal. (With the 0.5bp floor, the 20Y catch-up move on 2 September is no longer flagged; run with `mad_floor=0.1` to see it return.) The unit error is also caught by Day 3's plausibility check; the duplicate, the source conflict and the off-snapshot value are only caught by Day 3, because they're about *metadata*, not values. Every one of the eight problems is now covered by at least one check.

Consequence flags: the day after each bad value (28 August for the 1Y, 4 September for the 10Y), plus the 20Y catch-up on 2 September if your floor is low. One fix is to replace flagged values with NaN and recompute changes from the last *good* value, so one bad point produces one flag. That is what a production monitor should do, and saying so shows you've thought about alert fatigue: a monitor that cries wolf twice for every problem soon gets ignored.
</details>

## 4.10 Timed drill (10 min, no notes)

1. Reindex `wide` onto the business-day calendar.
2. Day-on-day changes in bp; list every move over 15bp.
3. A 5-day trailing moving average of the 2Y, with no look-ahead.
4. Robust z-score with a shifted 10-day window and a 0.5bp floor; list scores above 6.
5. Find tenors with 5 identical values in a row.
6. Weekly Friday closes for 2Y and 10Y.

## Day 4 checklist

- [ ] Why reindex onto a full calendar before taking differences?
- [ ] What's the difference between `diff` and `pct_change`, and which do you use for rates?
- [ ] What is look-ahead bias, and where can it creep in (`shift(-1)`, `center=True`, unshifted statistics)?
- [ ] Why is a robust z-score better than an ordinary one for monitoring?
- [ ] Why does MAD need a floor?
- [ ] How do you find consecutive runs with `cumsum`?
- [ ] What does `merge_asof` do, and why use `tolerance`?
- [ ] Why store timestamps in UTC?
- [ ] Why does one bad point create two jump flags, and how would you stop that?

---
# Day 5: Analysis, P&L Attribution and Production Code

**Goal:** turn clean data into analysis an interviewer will respect: PCA of curve moves, regression, value-at-risk, and P&L attribution with a root-cause investigation of every break. Then package the work as production-style code. This day pulls the whole week together and mirrors the job itself: find abnormal behaviour, explain it, and automate the analysis.
**Time:** about 2.5 hours.

## Warm-up (5 min, from memory, in `day5.ipynb`)

1. Write the robust z-score formula, and explain why the statistics are shifted by one day.
2. Write the `cumsum` trick for finding runs of unchanged values.
3. Name the three checks in yesterday's `health_check` and one problem each catches.

## 5.1 Move your code into a module (15 min)

Notebooks are for exploring; reusable logic belongs in a `.py` file you import. It's also what an interviewer means by "production-facing". Create `curvetools.py` in `~/pandas-course` with the functions from Days 2–4:

```python
# file: curvetools.py
"""Loading, reshaping and checking end-of-day SOFR OIS curves."""
import glob

import numpy as np
import pandas as pd

TENORS = ["ON", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


def load_curves(folder: str = "lab/data/curves") -> pd.DataFrame:
    """Read every daily curve file into one long DataFrame."""
    files = sorted(glob.glob(f"{folder}/sofr_ois_*.csv"))
    if not files:
        raise FileNotFoundError(f"no curve files in {folder}")
    return pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)


def to_wide(df: pd.DataFrame, calendar: pd.DatetimeIndex, source: str = "BBG") -> pd.DataFrame:
    """Date x tenor matrix for one source, reindexed onto the full business-day calendar."""
    sub = df[(df["source"] == source) & (df["snapshot_time"] == "16:00")]
    if sub.duplicated(["date", "tenor"]).any():
        raise ValueError("duplicate (date, tenor) rows for a single source")
    wide = sub.pivot(index="date", columns="tenor", values="rate_pct")
    wide = wide.reindex(index=calendar, columns=TENORS)
    wide.index.name = "date"
    return wide


def robust_z(chg: pd.DataFrame, window: int = 10, floor: float = 0.5) -> pd.DataFrame:
    med = chg.rolling(window, min_periods=5).median().shift(1)
    mad = (chg - med).abs().rolling(window, min_periods=5).median().shift(1)
    return (chg - med) / (1.4826 * mad.clip(lower=floor))


def health_check(wide: pd.DataFrame, z_threshold: float = 6, stale_len: int = 5) -> pd.DataFrame:
    """One row per issue: date, tenor, check, detail."""
    chg = wide.diff() * 100
    rz = robust_z(chg)
    out = []

    j = rz.stack()
    for (d, t), score in j[j.abs() > z_threshold].items():
        out.append((d, t, "jump", f"{chg.loc[d, t]:+.1f}bp, robust z {score:+.1f}"))

    unchanged = wide.diff().eq(0)
    for t in wide.columns:
        u = unchanged[t]
        run_id = (~u).cumsum()
        run_len = u.groupby(run_id).cumsum()
        long_runs = run_id[run_len >= stale_len - 1].unique()
        for d in u.index[u & run_id.isin(long_runs)]:
            out.append((d, t, "stale", f"unchanged at {wide.loc[d, t]}"))

    m = wide.isna().stack()
    for d, t in m[m].index:
        out.append((d, t, "missing", "no value"))

    return (pd.DataFrame(out, columns=["date", "tenor", "check", "detail"])
              .sort_values(["date", "tenor", "check"]).reset_index(drop=True))
```

Two deliberate changes from Day 4: `to_wide` **raises an error** on duplicates instead of letting `pivot` fail obscurely, and it **excludes off-snapshot rows**, so the 21:00 value becomes a missing value rather than silently entering the analysis. Both are policy decisions; in an interview, say they are, and say who should own them.

Now, in the notebook:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from curvetools import TENORS, load_curves, to_wide, health_check, robust_z
pd.set_option("display.width", 120)

calendar = pd.DatetimeIndex(pd.read_csv("lab/ref/business_days.txt", header=None)[0], name="date")
calendar = pd.to_datetime(calendar)
df = load_curves()
wide = to_wide(df, calendar)
h = health_check(wide)
h["check"].value_counts()
```

If you edit `curvetools.py` later, restart the notebook kernel (or use `%load_ext autoreload` then `%autoreload 2`), because Python only imports a module once.

## 5.2 Build a clean curve (20 min)

Turn the flags into a mask of **bad values**, then fill the gaps.

One subtlety from Day 4: a bad print creates two jump flags, the bad day and the reversal the next day. Only the first is a bad value; the second day's rate is fine. Mask a jump only when the next day reverses it:

```python
def flag_frame(issues, check, like):
    sel = issues[issues["check"] == check]
    return (pd.crosstab(sel["date"], sel["tenor"])
              .reindex(index=like.index, columns=like.columns, fill_value=0)
              .gt(0))

chg_raw = wide.diff() * 100
jump = flag_frame(h, "jump", wide)
stale = flag_frame(h, "stale", wide)

reversed_next = jump & jump.shift(-1, fill_value=False) & (np.sign(chg_raw) != np.sign(chg_raw.shift(-1)))
bad = reversed_next | stale
bad.stack()[lambda s: s].index.tolist()
```

You should see the 1Y on 27 August, the 10Y on 3 September, and the 20Y from 25 August to 1 September.

```python
clean = wide.mask(bad)                           # bad values become NaN
filled = clean.interpolate(limit_area="inside")  # straight line in time between good values
filled.isna().sum().sum()                        # 0
chg = filled.diff() * 100
```

`limit_area="inside"` only fills gaps that have good data on both sides; it never extrapolates past the first or last value.

**Be honest about what filling does.** Interpolation invents values. For the missing day it's probably close to the truth; for an eight-day stale run it's a guess that smooths over a week of real movement. Keep `clean` (with NaN) alongside `filled`, and record which cells were filled. For risk and P&L, many teams prefer another source or a model-based reconstruction over a straight line.

```python
was_filled = wide.isna() | bad
was_filled.sum()[lambda s: s > 0]
```

### Practice 1

1. Plot the 10Y and 20Y, raw and filled, on one chart. Where do they differ?
2. An alternative fill is across tenors on the same day (interpolating along the curve). Implement it for the missing 7Y on 12 August using `filled.T.interpolate()` on the raw-with-NaN frame. Which method gives a more plausible 7Y, and why might it depend on the gap?

<details><summary>Answer</summary>

```python
ax = wide[["10Y", "20Y"]].plot(style="--", alpha=0.6, figsize=(9, 4))
filled[["10Y", "20Y"]].plot(ax=ax)
ax.legend(["10Y raw", "20Y raw", "10Y filled", "20Y filled"])

along_curve = clean.T.interpolate().T            # interpolate across tenor columns, by position
along_curve.loc["2026-08-12", "7Y"], filled.loc["2026-08-12", "7Y"]
```

Interpolating across columns treats tenors as evenly spaced, which they aren't (5Y, 7Y, 10Y); a proper version interpolates on maturity in years. For one missing tenor, the same day's neighbours are usually better: they share the day's market move. For a missing *day*, interpolating in time is the only option within one source.
</details>

## 5.3 PCA: what drives curve moves (25 min)

Principal component analysis finds the few independent patterns that explain most of the variation in daily curve changes. For yield curves the first three are classically **level**, **slope** and **curvature**.

With NumPy, PCA is an eigen-decomposition of the covariance matrix:

```python
def pca(changes: pd.DataFrame):
    X = changes.dropna().to_numpy()
    X = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))   # eigh: for symmetric matrices
    order = np.argsort(vals)[::-1]                          # largest first
    explained = vals[order] / vals.sum()
    loadings = pd.DataFrame(vecs[:, order[:3]], index=changes.columns, columns=["PC1", "PC2", "PC3"])
    return explained, loadings

explained, loadings = pca(chg)
explained[:3].round(3)
loadings.round(2).T
```

PC1 explains about 90% of daily variation, and its loadings are all positive and similar: the whole curve moving up or down together, a **level** shift. That's typical for rates. (The sign of an eigenvector is arbitrary; `-PC1` is equally valid.)

Now the important experiment. Run the same PCA on the **raw** data:

```python
explained_raw, loadings_raw = pca(chg_raw)
explained_raw[:3].round(3)
loadings_raw.round(2).T
```

PC1 now "explains" over 98% of the variation, and it is almost entirely the 1Y. PC2 is almost entirely the 10Y. The "dominant risk factors" of this curve are two data errors. Any model trained on this data (a risk model, a hedge ratio, a signal) would be fitted to garbage while looking statistically impressive. This is the single best example to give in an interview of **why data quality comes before modelling**.

```python
fig, ax = plt.subplots(figsize=(8, 3.5))
loadings.plot(ax=ax, marker="o")
ax.set_xticks(range(len(TENORS)), TENORS)
ax.axhline(0, color="grey", lw=0.5)
ax.set_title("PCA loadings of daily SOFR OIS changes (clean data)")
plt.tight_layout()
```

With only 29 days, PC2 and PC3 are noisy, and in this synthetic dataset they won't look like textbook slope and curvature. On real data, with a few years of history, they usually do. Saying that is better than over-interpreting.

## 5.4 Regression and the look-ahead trap (20 min)

How does the 10Y move when the 2Y moves?

```python
x, y = chg["2Y"], chg["10Y"]
ok = x.notna() & y.notna()
A = np.column_stack([np.ones(ok.sum()), x[ok]])
beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
beta.round(3), np.corrcoef(x[ok], y[ok])[0, 1].round(3)
```

A slope just above 1 and a correlation above 0.9: the two tenors move nearly one-for-one. On the raw data:

```python
xr, yr = chg_raw["2Y"], chg_raw["10Y"]
okr = xr.notna() & yr.notna()
np.polyfit(xr[okr], yr[okr], 1).round(3), np.corrcoef(xr[okr], yr[okr])[0, 1].round(3)
```

The raw correlation collapses to about 0.26 because of one bad point. Two outliers out of about two dozen observations destroy a relationship that is really very strong. (`np.polyfit(x, y, 1)` returns `[slope, intercept]`, in that order.)

**Rolling regression and look-ahead.** A relative-value signal compares the 10Y with what the 2Y implies, using a fitted beta. Fitting beta on the **whole sample** and then using it on each day in the past uses information from the future. A rolling, trailing fit doesn't:

```python
win = 10
b = x.rolling(win).cov(y) / x.rolling(win).var()       # beta from the last 10 days, up to and including today
a = y.rolling(win).mean() - b * x.rolling(win).mean()
roll_beta, roll_alpha = b.shift(1), a.shift(1)         # use only what was known BEFORE today
resid = y - (roll_alpha + roll_beta * x)
pd.DataFrame({"roll_beta": roll_beta, "resid_bp": resid}).dropna().round(2).head()
```

The `shift(1)` means each day's beta is estimated without that day. In an interview about a backtest, the first question is often "was the beta or z-score estimated in-sample?" Have this answer ready.

## 5.5 Value-at-risk, properly this time (15 min)

The book's PV01:

```python
pos = pd.read_csv("lab/data/positions.csv").set_index("tenor")["pv01_usd"].reindex(TENORS)
pos
```

**Historical P&L from curve moves:** each day's change vector times PV01.

```python
hist_pnl = chg.dropna() @ pos            # DataFrame @ Series → one P&L per day
hist_pnl.describe().round(0)
```

**Historical VaR** is a percentile of these outcomes; **parametric (Monte Carlo) VaR** simulates from the covariance:

```python
hist_var = -np.percentile(hist_pnl, 1)
rng = np.random.default_rng(42)
cov = np.cov(chg.dropna().to_numpy(), rowvar=False)
sim = rng.multivariate_normal(np.zeros(len(TENORS)), cov, size=100_000)
mc_var = -np.percentile(sim @ pos.to_numpy(), 1)
round(hist_var), round(mc_var)
```

The two are similar here, but be candid about the limits: the 1st percentile of 29 observations is essentially the worst day; a normal distribution understates fat tails; and six weeks of a quiet market says little about a stressed one. That honesty is worth more in an interview than the number.

Compute the same VaR on the raw changes and see what the data errors do to it. (Expect it to be absurd.)

## 5.6 P&L attribution and root-cause analysis (30 min)

This is the closest exercise to the job: "production P&L diverged from expectations; find out why."

**Explained P&L** is what the curve moves and the book's PV01 imply. **Reported P&L** comes from the risk system. The difference is the **unexplained** P&L, or "break".

```python
rep = pd.read_csv("lab/data/pnl_reported.csv", parse_dates=["date"]).set_index("date")["pnl_usd"]

recon = pd.DataFrame({
    "reported": rep,
    "explained_raw": (chg_raw * pos).sum(axis=1, min_count=len(TENORS)),
    "explained_clean": (chg * pos).sum(axis=1, min_count=len(TENORS)),
})
recon["break_raw"] = recon["reported"] - recon["explained_raw"]
recon["break_clean"] = recon["reported"] - recon["explained_clean"]
recon.round(0)
```

`min_count=12` makes the sum NaN unless all 12 tenors are present, so a missing tenor doesn't silently count as zero P&L.

Read the raw columns first. Breaks of nearly a million dollars on 27–28 August and half a million on 3–4 September, plus several days that can't be explained at all because of missing data. With raw data, P&L explain is useless.

Now the clean columns. First, find out which days have data issues. A daily P&L uses two curves, today's and yesterday's, so an issue on either day counts:

```python
issue_today = was_filled.any(axis=1)
issue_either = issue_today | issue_today.shift(1, fill_value=False)
issue_either.sum()
```

**Set the threshold from days you can trust.** If you estimate "normal" break size from all days, the data-issue days inflate it and hide real problems. Use the robust spread of breaks on clean days only:

```python
b = recon["break_clean"]
clean_days = b[~issue_either & b.notna()]
threshold = 5 * 1.4826 * (clean_days - clean_days.median()).abs().median()
round(threshold), len(clean_days)
```

On clean days the breaks are only a few hundred dollars (the risk system's own small differences), so the threshold is about 1,400.

```python
diag = recon[["break_raw", "break_clean"]].assign(
    data_issue=issue_either,
    big_break=b.abs() > threshold,
)
diag[diag["big_break"] | diag["data_issue"]].round(0)
```

**Root cause.** Classify every day:

- **Big break, no data issue: 7 August (about +40,000) and 9 September (about −25,000).** The curve data is clean on both days, and on every other clean day explained P&L matches reported to within a few hundred dollars. So the break is in the reported number, not in our explanation of it: a trade booking, an amendment, a fee, or a missed position change. The next step is the trade blotter for those dates, not the market data.
- **Huge raw break, small clean break: 27–28 August and 3–4 September.** Breaks of up to a million dollars were entirely caused by the bad 1Y and 10Y values. Fixing the inputs fixed the P&L.
- **Moderate break on a day with estimated data** (roughly 3,000–13,000 on the missing day, the missing and blank values, the stale 20Y run and the 21:00 2Y): about the size you'd expect from interpolation error on a book with up to 15,000 of PV01 per bp. Report these as "explained within data-quality tolerance, curve values estimated", and confirm with a second source if the amounts matter. Don't present them as clean, and don't escalate them as booking errors.

That three-way split, with evidence for each, is exactly what "root cause analysis" means in the job description.

### Practice 2

1. Make a bar chart of `break_clean` with the threshold drawn as horizontal lines, and colour the bars by whether the day had a data issue.
2. Suppose the 2Y PV01 were mis-booked as −90,000 instead of −9,000 from some date onwards. How would the break series look? (Think about it: it would correlate with the 2Y's daily move.) Write the regression that would detect it.

<details><summary>Answer</summary>

```python
colors = np.where(diag["data_issue"], "tab:orange", "tab:blue")
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(diag.index, diag["break_clean"].fillna(0), color=colors)
for s in (1, -1):
    ax.axhline(s * threshold, color="red", ls="--", lw=1)
ax.set_title("Unexplained P&L (orange: data issue on the day or the day before)")
plt.tight_layout()

ok = recon["break_clean"].notna() & chg["2Y"].notna()
np.polyfit(chg.loc[ok, "2Y"], recon.loc[ok, "break_clean"], 1)   # slope is small here
```

A wrong PV01 produces breaks proportional to that tenor's move, so regressing the breaks on each tenor's change finds it: a slope near the PV01 error (here it would be about −81,000 per bp) and a high R². With correct positions, as here, the slopes are small: around 1,800 per bp for the 2Y, driven mostly by the interpolated days, against the roughly −81,000 a mis-booking would produce. This "regress the residual on the risk factors" idea is a powerful general diagnostic.
</details>

## 5.7 Writing it like production code (15 min)

Interviewers at a systematic fund will look at *how* you write pandas, not just whether it works. Habits to show:

- **Functions with clear inputs and outputs**, in a module, not 200 lines of notebook cells.
- **Type hints and docstrings**, as in `curvetools.py`.
- **Fail loudly on broken assumptions**: `raise` on duplicates, `validate=` on merges, assertions on shapes.
- **No magic numbers buried in code**: thresholds as parameters with defaults.
- **Vectorised operations**, no `iterrows`; `apply` only where unavoidable.
- **Tests.** Even a few assertions on a tiny hand-made input catch most regressions:

```python
def test_health_check_finds_stale_and_jump():
    idx = pd.bdate_range("2026-01-05", periods=20)
    rng = np.random.default_rng(0)
    w = pd.DataFrame({"2Y": 3.5 + rng.normal(0, 0.02, 20).cumsum(),
                      "10Y": 4.0 + rng.normal(0, 0.02, 20).cumsum()}, index=idx)
    w.iloc[12:18, 0] = w.iloc[12, 0]          # stale 2Y for 6 values
    w.iloc[15, 1] += 0.5                      # 50bp bad print on the 10Y
    issues = health_check(w)
    assert set(issues.loc[issues.check == "stale", "tenor"]) == {"2Y"}
    assert (issues.loc[issues.check == "jump", "tenor"] == "10Y").any()
    assert idx[15] in set(issues.loc[issues.check == "jump", "date"])

test_health_check_finds_stale_and_jump()
print("test passed")
```

Put tests like this in `test_curvetools.py` and run them with `pytest` (`pip install pytest`, then `pytest` in the folder). Building a small synthetic case where you *know* the right answer is the most convincing way to show a monitor works.

**Performance habits** for large data:

```python
df.memory_usage(deep=True).sum()
df.astype({"tenor": "category", "source": "category", "snapshot_time": "category"}).memory_usage(deep=True).sum()
```

Categoricals for repeated strings, `float32` where precision allows, reading only needed columns (`usecols=`), Parquet instead of CSV, and filtering before joining. Beyond a few gigabytes, mention **polars** (multi-threaded, lazy) or doing the heavy aggregation in SQL first; the job ad names polars, so a short comparison is in Appendix D.

## 5.8 Capstone: the daily report (25 min)

Write a function `daily_report(date)` that, for one date, prints a short plain-English report a trading lead could read in 30 seconds:

1. Data health: issues on that date (from `health_check` and Day 3's metadata checks), or "all checks passed".
2. The curve: 2s10s and 5s30s slopes in bp, and the change since the previous day.
3. P&L: reported, explained (clean), break, and whether the break is within threshold.
4. A one-line verdict: "clean", "explained within data tolerance", or "unexplained break on clean data: investigate bookings".

Run it for 7 August, 27 August, 3 September and 9 September. Then write, in your own words, five sentences summarising the six weeks for a non-technical manager. That written summary is practice for the "complete and clear reports" line in the job ad, and it's a strong thing to talk through in an interview.

<details><summary>Answer (one possible version)</summary>

```python
def daily_report(date):
    d = pd.Timestamp(date)
    lines = [f"Report for {d.date()}"]
    iss = h[h["date"] == d]
    meta = df[(df["date"] == d) & ((df["source"] != "BBG") | (df["snapshot_time"] != "16:00"))]
    if iss.empty and meta.empty:
        lines.append("  Data: all checks passed")
    for _, r in iss.iterrows():
        lines.append(f"  Data: {r['check']} {r['tenor']} ({r['detail']})")
    for _, r in meta.iterrows():
        lines.append(f"  Data: metadata issue {r['tenor']} source={r['source']} snapshot={r['snapshot_time']}")

    s210 = (filled.loc[d, "10Y"] - filled.loc[d, "2Y"]) * 100
    s530 = (filled.loc[d, "30Y"] - filled.loc[d, "5Y"]) * 100
    prev = filled.index[filled.index.get_loc(d) - 1]
    d210 = s210 - (filled.loc[prev, "10Y"] - filled.loc[prev, "2Y"]) * 100
    lines.append(f"  Curve: 2s10s {s210:.1f}bp ({d210:+.1f}), 5s30s {s530:.1f}bp")

    r = recon.loc[d]
    within = abs(r["break_clean"]) <= threshold
    lines.append(f"  P&L: reported {r['reported']:,.0f}, explained {r['explained_clean']:,.0f}, "
                 f"break {r['break_clean']:,.0f} ({'within' if within else 'OUTSIDE'} threshold)")

    data_issue = was_filled.loc[d].any() or was_filled.loc[prev].any()
    if within:
        verdict = "clean" if not data_issue else "clean P&L, but some curve values were estimated"
    elif data_issue:
        verdict = "explained within data tolerance: curve values estimated, confirm with a second source"
    else:
        verdict = "unexplained break on clean data: investigate bookings"
    lines.append(f"  Verdict: {verdict}")
    print("\n".join(lines))

for d in ["2026-08-07", "2026-08-27", "2026-09-03", "2026-09-09"]:
    daily_report(d)
    print()
```

`iterrows` is fine here: it loops over a handful of issue rows to format text, not over data for calculation.
</details>

## Day 5 checklist

- [ ] Why move notebook code into a module, and what makes code "production-style"?
- [ ] Why mask only the first of a jump-and-reversal pair?
- [ ] What are the risks of interpolating missing market data?
- [ ] How do you compute PCA with NumPy, and what do the first three components usually mean for rates?
- [ ] What did the raw-data PCA teach you?
- [ ] How can a full-sample regression introduce look-ahead bias?
- [ ] What are the limits of historical and Monte Carlo VaR on six weeks of data?
- [ ] How do you separate P&L breaks caused by data from breaks caused by bookings?
- [ ] How would you detect a mis-booked PV01 from the break series?
- [ ] How do you test a data-health monitor?

---
# Appendix A: Interview Quick-Fire Questions

Cover the right-hand column and answer aloud.

| Question | Short answer |
|---|---|
| Why is NumPy faster than Python lists? | Contiguous, typed memory and compiled loops; operations run on whole arrays at once. |
| What is broadcasting? | Combining arrays of different shapes by stretching dimensions of size 1, without copying. |
| `axis=0` vs `axis=1` | `axis=0` collapses rows (one result per column); `axis=1` collapses columns (one per row). |
| Series vs DataFrame | A labelled 1-D array vs a table of Series sharing an index. |
| What does index alignment mean? | Operations match on labels, not positions; unmatched labels give NaN. |
| `loc` vs `iloc` | Labels and booleans (end-inclusive slices) vs integer positions (end-exclusive). |
| Why is `df[mask]["col"] = x` wrong? | Chained assignment modifies a temporary copy; use `df.loc[mask, "col"] = x`. |
| What is Copy-on-Write? | pandas 3 default: derived objects behave as copies, so changes never leak back to the original. |
| `count` vs `size` in groupby | `count` ignores NaN; `size` counts rows. |
| `agg` vs `transform` | `agg` returns one row per group; `transform` returns one value per original row. |
| `merge` vs `concat` vs `join` | Merge aligns by key columns; concat stacks; join merges on the index. |
| How can a merge go wrong silently? | Duplicate keys multiply rows; use `validate=` and check row counts. |
| `pivot` vs `pivot_table` | `pivot` fails on duplicates; `pivot_table` aggregates them, which can hide problems. |
| Why avoid `apply` and `iterrows`? | They're Python loops; vectorised methods are 10–100× faster. |
| How do you detect outliers robustly? | Rolling median and MAD (with a floor), trailing window, shifted by one day. |
| What is look-ahead bias? | Using information that wasn't available at the time: `shift(-1)`, centred windows, full-sample fits. |
| How do you find consecutive runs? | `run_id = (~cond).cumsum()`, then `cond.groupby(run_id).cumsum()`. |
| What does `merge_asof` do? | Joins each row to the latest row of another table at or before its time; use `tolerance`. |
| Why reindex a time series onto a calendar? | So gaps become explicit NaN and changes aren't silently multi-day. |
| How would you fill a missing price? | Prefer flagging and a second source; interpolation is an estimate; never fill with zero. |
| PCA on yield curves gives what? | Level (~90% of variance), slope, curvature, on a long, clean history. |
| How do you decide if a P&L break is a data issue? | Check data flags on both days used; estimate normal break size from clean days; breaks on clean data point to bookings. |
| How would you make a monitor production-ready? | Functions, parameters, tests on synthetic cases, loud failures, logging, scheduling, alert suppression. |
| When is pandas the wrong tool? | Data too large for memory or speed: polars, SQL, or a distributed engine. |

# Appendix B: Common Errors and Fixes

| Symptom | Usual cause | Fix |
|---|---|---|
| `ValueError: Index contains duplicate entries, cannot reshape` | Duplicate keys in `pivot` | Find them with `df[df.duplicated(keys, keep=False)]`; decide which wins |
| `ChainedAssignmentError` or `SettingWithCopyWarning` | `df[mask]["col"] = x` | `df.loc[mask, "col"] = x`; `.copy()` subsets you intend to modify |
| Unexpected NaN after arithmetic | Labels don't align | Check indexes; `.reset_index(drop=True)` or `.to_numpy()` if positions are meant |
| Row count grows after merge | Duplicate keys on the "one" side | `validate="many_to_one"`; deduplicate first |
| `KeyError` on a column that "exists" | Trailing spaces or different case in the header | `df.columns = df.columns.str.strip()` |
| Dates sort as text, or `.dt` fails | Column is still a string | `pd.to_datetime`, or `parse_dates=` in `read_csv` |
| Tenors sort as `10Y, 1M, 1Y` | Text sort | Ordered `pd.Categorical` |
| `ValueError: The truth value of a Series is ambiguous` | `and`/`or`/`if` on a Series | `&`, `|`, `~` with brackets; `.any()` / `.all()` |
| A comparison returns False for missing values | `NaN == x` is always False | `isna()` / `notna()` explicitly |
| `inf` in a z-score | Zero dispersion in the denominator | Floor it: `mad.clip(lower=...)` |
| Rolling results all NaN at the start | Window not full | `min_periods=` |
| Monitor flags look too good in a backtest | Look-ahead | Trailing windows, `.shift(1)` statistics, no `center=True` |
| `np.cov` returns the wrong shape | Variables in columns but `rowvar` left True | `np.cov(X, rowvar=False)` |
| Module changes not picked up in the notebook | Python caches imports | Restart the kernel, or `%autoreload 2` |

# Appendix C: Cheat Sheet

```text
# NumPy
a = np.array([...]); a.shape; a.dtype; np.arange; np.linspace; np.zeros((r, c))
a[mask]; a[(a > 1) & (a < 2)]; np.where(cond, x, y); a.argmax(); np.diff(a)
M.mean(axis=0); M[:, j]; v[:, None]; A @ b; np.isnan; np.nanmean
rng = np.random.default_rng(42); np.cov(X, rowvar=False); np.linalg.eigh; np.linalg.lstsq

# Load and inspect
df = pd.concat((pd.read_csv(f, parse_dates=["date"]) for f in files), ignore_index=True)
df.head(); df.info(); df.describe(); df.isna().sum(); df["col"].value_counts()

# Select and modify
df.loc[cond, ["a", "b"]]; df.iloc[:5, :2]; df.query("a > 1"); df["a"].isin([...])
df.loc[cond, "a"] = np.nan; df.assign(b=lambda d: d["a"] * 100)

# Group, reshape, merge
df.groupby("k").agg(avg=("v", "mean"), n=("v", "size"))
df.groupby("k")["v"].transform("median")
df.pivot(index="date", columns="tenor", values="v"); wide.melt(...); wide.stack()
df.duplicated(keys, keep=False); a.merge(b, on="k", how="left", indicator=True, validate="many_to_one")

# Time series
wide = wide.reindex(calendar); wide.diff() * 100; wide.shift(1)
wide.rolling(10, min_periods=5).median().shift(1); wide.resample("W-FRI").last()
pd.merge_asof(trades, quotes, on="time", tolerance=pd.Timedelta("5s"))
run_id = (~cond).cumsum(); cond.groupby(run_id).cumsum()
```

# Appendix D: pandas to polars, a quick translation

The Voleon job ad asks for polars. The concepts you practised all carry over; the syntax is expression-based. Install with `pip install polars`.

```python
import polars as pl

pdf = pl.read_csv("lab/data/curves/*.csv", try_parse_dates=True)    # globs work directly
pdf.shape
```

| Task | pandas | polars |
|---|---|---|
| Filter | `df[df["tenor"] == "10Y"]` | `pdf.filter(pl.col("tenor") == "10Y")` |
| Select columns | `df[["date", "rate_pct"]]` | `pdf.select("date", "rate_pct")` |
| New column | `df.assign(bp=df["rate_pct"] * 100)` | `pdf.with_columns((pl.col("rate_pct") * 100).alias("bp"))` |
| Group and aggregate | `df.groupby("tenor")["rate_pct"].mean()` | `pdf.group_by("tenor").agg(pl.col("rate_pct").mean())` |
| Group-wise column (`transform`) | `df.groupby("tenor")["rate_pct"].transform("median")` | `pl.col("rate_pct").median().over("tenor")` |
| Sort | `df.sort_values(["tenor", "date"])` | `pdf.sort("tenor", "date")` |
| Day-on-day change per tenor | `df.groupby("tenor")["rate_pct"].diff()` | `pl.col("rate_pct").diff().over("tenor")` (after sorting by date) |
| Duplicates | `df[df.duplicated(["date", "tenor"], keep=False)]` | `pdf.filter(pl.struct("date", "tenor").is_duplicated())` |
| Pivot | `df.pivot(index="date", columns="tenor", values="rate_pct")` | `pdf.pivot("tenor", index="date", values="rate_pct")` |
| As-of join | `pd.merge_asof(a, b, on="time")` | `a.join_asof(b, on="time")` |
| Lazy pipeline | n/a | `pl.scan_csv(...).filter(...).collect()` |

Try the Day 3 duplicate check and the Day 4 day-on-day changes in polars:

```python
pdf.filter(pl.struct("date", "tenor").is_duplicated())

(pdf.filter(pl.col("source") == "BBG")
    .sort("tenor", "date")
    .with_columns((pl.col("rate_pct").diff().over("tenor") * 100).alias("chg_bp"))
    .filter(pl.col("chg_bp").abs() > 15)
    .select("date", "tenor", "chg_bp"))
```

Key differences to mention in an interview: polars has no index (so no alignment surprises and no `loc`/`iloc`), it's multi-threaded and usually much faster on large data, and its lazy mode optimises the whole query before running it. pandas remains the default in most research code and has the larger ecosystem.
