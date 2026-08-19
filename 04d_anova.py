# EX.NO: 4D - ANOVA on Diabetes Datasets
# AIM: Perform one-way ANOVA to compare feature distributions across groups

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

np.random.seed(42)
n = 200

uci = pd.DataFrame({
    'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
    'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
    'Outcome':       np.random.randint(0, 2, n)
})

pima = pd.DataFrame({
    'Glucose':       np.random.normal(136, 32, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(82, 21, n).clip(40, 130).astype(int),
    'BMI':           np.random.normal(32.5, 6.9, n).clip(15, 60),
    'Outcome':       np.random.randint(0, 2, n)
})

print("=" * 60)
print("  ANOVA: UCI Diabetes vs Pima Indians")
print("=" * 60)
for col in ['Glucose', 'BloodPressure', 'BMI']:
    f_stat, p_val = f_oneway(uci[col], pima[col])
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"\n--- {col} ---")
    print(f"  F-Statistic: {f_stat:.4f}")
    print(f"  P-Value    : {p_val:.4f}")
    print(f"  Result     : {sig} (alpha=0.05)")

print("\n\n--- Within-dataset ANOVA (UCI: Diabetic vs Non-diabetic) ---")
for col in ['Glucose', 'BloodPressure', 'BMI']:
    diabetic     = uci[uci['Outcome'] == 1][col]
    non_diabetic = uci[uci['Outcome'] == 0][col]
    f_stat, p_val = f_oneway(diabetic, non_diabetic)
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"  {col}: F={f_stat:.4f}, P={p_val:.4f} -> {sig}")
