# EX.NO: 4C - T-Test on Diabetes Datasets
# AIM: Perform independent t-test to compare UCI and Pima datasets

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

np.random.seed(42)
n = 200

uci = pd.DataFrame({
    'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
    'BMI':           np.random.normal(31, 7.8, n).clip(15, 60)
})

pima = pd.DataFrame({
    'Glucose':       np.random.normal(136, 32, n).clip(50, 200).astype(int),
    'BloodPressure': np.random.normal(82, 21, n).clip(40, 130).astype(int),
    'BMI':           np.random.normal(32.5, 6.9, n).clip(15, 60)
})

print("=" * 60)
print("  Independent T-Test: UCI Diabetes vs Pima Indians")
print("=" * 60)

for col in ['Glucose', 'BloodPressure', 'BMI']:
    t_stat, p_val = ttest_ind(uci[col], pima[col], equal_var=False)
    significance = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"\n--- {col} ---")
    print(f"  UCI Mean   : {uci[col].mean():.2f}")
    print(f"  Pima Mean  : {pima[col].mean():.2f}")
    print(f"  T-Statistic: {t_stat:.4f}")
    print(f"  P-Value    : {p_val:.4f}")
    print(f"  Result     : {significance} (alpha=0.05)")

print("\n--- One-Sample T-Test (UCI Glucose, H0: mean=120) ---")
t_stat, p_val = ttest_1samp(uci['Glucose'], 120)
print(f"  T-Statistic: {t_stat:.4f}")
print(f"  P-Value    : {p_val:.4f}")
print(f"  Result     : {'REJECT H0' if p_val < 0.05 else 'FAIL TO REJECT H0'}")
