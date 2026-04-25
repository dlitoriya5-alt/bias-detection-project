import pandas as pd

def check_bias(df, preds, sensitive_col, target):
    df = df.copy()

    df = df.iloc[:len(preds)]
    df["prediction"] = preds

    groups = df.groupby(sensitive_col)["prediction"].mean()

    result = {}

    for group, value in groups.items():
        result[str(group)] = round(value, 3)

    if len(groups) >= 2:
        diff = abs(groups.iloc[0] - groups.iloc[1])
    else:
        diff = 0

    result["bias_difference"] = round(diff, 3)

    if diff > 0.2:
        result["status"] = "⚠️ Bias Detected"
    else:
        result["status"] = "✅ Fair"

    return result