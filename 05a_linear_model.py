# EX.NO: 5A - Building and Validating Linear Regression Models
# AIM: Build, train and evaluate linear regression models on diabetes datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

np.random.seed(42)
n = 200

def gen_dataset(seed):
    np.random.seed(seed)
    return pd.DataFrame({
        'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
        'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
        'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
        'Age':           np.random.randint(21, 80, n)
    })

uci  = gen_dataset(42)
pima = gen_dataset(99)

features = ['Glucose', 'BloodPressure', 'BMI']
target   = 'Age'

for name, df in [("UCI Diabetes", uci), ("Pima Indians", pima)]:
    X = df[features]; y = df[target]
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xt, yt)
    y_pred = model.predict(Xte)

    r2  = r2_score(yte, y_pred)
    mse = mean_squared_error(yte, y_pred)
    mae = mean_absolute_error(yte, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name} - Linear Regression Model")
    print(f"{'='*50}")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MSE      : {mse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {np.sqrt(mse):.4f}")

    # Actual vs Predicted Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(yte, y_pred, color='blue', alpha=0.5, s=30)
    plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--', lw=2)
    plt.xlabel('Actual Age'); plt.ylabel('Predicted Age')
    plt.title(f'{name}: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'linear_model_{name.split()[0]}.png')
    plt.show()
    print(f"  Saved: linear_model_{name.split()[0]}.png")
