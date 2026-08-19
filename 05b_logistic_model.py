# EX.NO: 5B - Building and Validating Logistic Regression Models
# AIM: Build and evaluate logistic regression models for diabetes prediction

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

np.random.seed(42)
n = 200

def gen_dataset(seed):
    np.random.seed(seed)
    return pd.DataFrame({
        'Glucose':       np.random.normal(137, 36, n).clip(50, 200).astype(int),
        'BloodPressure': np.random.normal(83, 19, n).clip(40, 130).astype(int),
        'BMI':           np.random.normal(31, 7.8, n).clip(15, 60),
        'Age':           np.random.randint(21, 80, n),
        'Outcome':       np.random.randint(0, 2, n)
    })

uci  = gen_dataset(42)
pima = gen_dataset(99)

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
target   = 'Outcome'

for name, df in [("UCI Diabetes", uci), ("Pima Indians", pima)]:
    X = df[features]; y = df[target]
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(Xt, yt)
    y_pred = model.predict(Xte)

    print(f"\n{'='*50}")
    print(f"  {name} - Logistic Regression")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(yte, y_pred):.4f}")
    print(f"  Precision : {precision_score(yte, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(yte, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(yte, y_pred, zero_division=0):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(yte, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(yte, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'{name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'confusion_{name.split()[0]}.png')
    plt.show()
    print(f"  Saved: confusion_{name.split()[0]}.png")
