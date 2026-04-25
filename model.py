from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

def train_model(df, target):
    df = df.copy()

    # ❌ Drop useless columns
    drop_cols = ["Name", "Ticket", "Cabin"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ❗ Handle missing values (IMPORTANT)
    df = df.fillna(0)

    # ✅ Convert ALL categorical columns properly
    df = pd.get_dummies(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, X_test, y_test, preds