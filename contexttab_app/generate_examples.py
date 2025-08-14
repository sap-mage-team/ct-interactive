# contexttab_app/generate_examples.py
"""
Utility to create three example Excel files with ~10% missing values
into ./examples. You don't have to run this (the app can synthesize
on the fly), but having files helps for demos.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes

EX_DIR = Path(__file__).parent / "examples"
EX_DIR.mkdir(exist_ok=True)

def inject_missing_random(df: pd.DataFrame, frac: float = 0.10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.random(df.shape) < frac
    out = df.copy().astype("object")
    for j, col in enumerate(out.columns):
        out.loc[m[:, j], col] = pd.NA
    return out

def make_iris():
    d = load_iris(as_frame=True)
    df = d.frame.rename(columns={"target": "target"})
    return inject_missing_random(df, 0.10, 42)

def make_wine():
    d = load_wine(as_frame=True)
    df = d.frame.rename(columns={"target": "target"})
    return inject_missing_random(df, 0.10, 43)

def make_diabetes():
    d = load_diabetes(as_frame=True)
    df = pd.concat([d.data, d.target.rename("target")], axis=1)
    return inject_missing_random(df, 0.10, 44)

for name, maker in [
    ("iris_example.xlsx", make_iris),
    ("wine_example.xlsx", make_wine),
    ("diabetes_example.xlsx", make_diabetes),
]:
    df = maker()
    p = EX_DIR / name
    df.to_excel(p, index=False)
    print(f"Saved {p} with shape {df.shape}")