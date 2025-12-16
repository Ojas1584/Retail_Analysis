#pip install pandas numpy scikit-learn statsmodels


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import Holt, SimpleExpSmoothing
import warnings
import io

warnings.filterwarnings("ignore")


input_file = "Jumbo & Company_ Attach % .xls"
df = pd.read_excel(input_file)


MONTH_COLS = ["Aug", "Sep", "Oct", "Nov", "Dec"]


def linear_trend_predict(y, step):
    X = np.arange(1, len(y) + 1).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return float(model.predict([[step]])[0])

def rolling_mean_predict(y, window=3):
    return float(np.mean(y[-window:]))

def holt_predict(y):
    if len(y) < 2:
        return np.mean(y)
    try:
        model = Holt(y, initialization_method="estimated").fit(optimized=True)
        return float(model.forecast(1)[0])
    except:
        try:
            ses = SimpleExpSmoothing(y, initialization_method="estimated").fit()
            return float(ses.forecast(1)[0])
        except:
            return np.mean(y)

def rmse_cv(y, model_fn):
    errors = []
    for i in range(3, len(y)):
        train = y[:i]
        actual = y[i]
        try:
            pred = model_fn(train, i + 1)
        except:
            pred = np.mean(train)
        errors.append((pred - actual) ** 2)
    return np.sqrt(np.mean(errors)) if errors else np.nan

def cap_value(x):
    return round(min(max(x, 0.0), 1.0), 3)


results = []
MAX_JUMP = 0.30  

for _, row in df.iterrows():

    y = row[MONTH_COLS].values.astype(float)

    if np.sum(y) == 0:
        results.append({
            "Branch": row["Branch"],
            "Store_Name": row["Store_Name"],
            **dict(zip(MONTH_COLS, y)),
            "Jan_Prediction": 0.0,
            "Volatility": 0.0,
            "Momentum": 0.0,
            "Confidence": "Low",
            "Store_Category": "New / Zero Performance"
        })
        continue

    volatility = np.std(y)
    momentum = y[-1] - y[0]

    pred_lm = linear_trend_predict(y, 6)
    pred_rm = rolling_mean_predict(y)
    pred_holt = holt_predict(y)


    rmse_lm = rmse_cv(y, linear_trend_predict)
    rmse_rm = rmse_cv(y, lambda a, b: rolling_mean_predict(a))
    rmse_holt = rmse_cv(y, lambda a, b: holt_predict(a))

    rmses = [rmse_rm, rmse_lm, rmse_holt]
    rmses = [r if not np.isnan(r) else 1e6 for r in rmses]

    inv = [1 / (r + 1e-9) for r in rmses]
    w_rm, w_lm, w_holt = [i / sum(inv) for i in inv]


    shrink = min(0.6, volatility / 0.30)
    w_rm = (1 - shrink) * w_rm + shrink * 0.6
    w_lm = (1 - shrink) * w_lm + shrink * 0.4

    total_w = w_rm + w_lm + w_holt
    w_rm, w_lm, w_holt = w_rm / total_w, w_lm / total_w, w_holt / total_w

    raw_pred = (
        w_rm * pred_rm +
        w_lm * pred_lm +
        w_holt * pred_holt
    )


    last_val = y[-1]
    if abs(raw_pred - last_val) > MAX_JUMP:
        raw_pred = last_val + np.sign(raw_pred - last_val) * MAX_JUMP

    final_pred = cap_value(raw_pred)


    confidence = "High"
    if volatility > 0.15:
        confidence = "Low"
    elif volatility > 0.08:
        confidence = "Medium"


    if final_pred >= 0.45 and volatility < 0.12:
        category = "Star Performer"
    elif momentum >= 0.15:
        category = "High Growth"
    elif final_pred < 0.20 and volatility < 0.10:
        category = "Consistent Laggard"
    elif final_pred < 0.20:
        category = "Volatile Laggard"
    else:
        category = "Stable / Average"

    results.append({
        "Branch": row["Branch"],
        "Store_Name": row["Store_Name"],
        **dict(zip(MONTH_COLS, y)),
        "Jan_Prediction": final_pred,
        "Volatility": round(volatility, 3),
        "Momentum": round(momentum, 3),
        "Confidence": confidence,
        "Store_Category": category
    })



final_df = pd.DataFrame(results)


for col in [
    'Jan_Prediction', 'Volatility', 'Momentum'
] + MONTH_COLS:
    if col in final_df.columns:
        final_df[col] = final_df[col].round(3)

final_df.to_csv("store_forecast_output_FINAL.csv", index=False)

print(" Forecast saved: store_forecast_output_FINAL.csv")
