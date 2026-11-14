# Linear regression

Compares linear, quadratic, and cubic fits on a two-feature dataset and prints train and test mean squared error.

## What it does

- Reads `x1`, `x2`, and `y` from the CSV.
- Holds out 20% of the rows with `random_state=2022`.
- Fits polynomial features of degree 1, 2, and 3 with scikit-learn's `LinearRegression`.
- Prints a small table of train and test MSE for the three fits.

## Layout

- `src/a1.py` — the three fits and the results table
- `data/a1_data.csv` — features `x1` and `x2`, response `y`

## Run

From the repository root:

```bash
pip install -r requirements.txt
python src/a1.py
```

## Stack

Python, pandas, NumPy, scikit-learn, Matplotlib. Written for MEEN 423.
