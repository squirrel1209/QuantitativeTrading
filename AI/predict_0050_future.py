import json
import pandas as pd
import numpy as np
import pickle
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.metrics import accuracy_score

# === 1. 載入未來資料 ===
with open("../output/0050_future.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["data"])
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.sort_index()

# === 2. 計算技術指標 ===
df["ma5"] = SMAIndicator(df["close"], window=5).sma_indicator()
df["ma10"] = SMAIndicator(df["close"], window=10).sma_indicator()
df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
df["next_close"] = df["close"].shift(-1)
df["label"] = (df["next_close"] > df["close"]).astype(int)
df = df.dropna()

# === 3. 載入訓練好的模型 ===
with open("xgb_model_0050.pkl", "rb") as f:
    model = pickle.load(f)

# === 4. 建立特徵並預測 ===
window_size = 10
features, labels, dates, closes, next_closes = [], [], [], [], []

for i in range(window_size, len(df) - 1):
    close_window = df["close"].values[i - window_size:i]
    volume_window = df["volume"].values[i - window_size:i]
    ma5 = df["ma5"].values[i]
    ma10 = df["ma10"].values[i]
    rsi = df["rsi"].values[i]

    feature = np.concatenate([close_window, volume_window, [ma5, ma10, rsi]])
    features.append(feature)
    labels.append(df["label"].values[i])
    dates.append(df.index[i])
    closes.append(df["close"].values[i])
    next_closes.append(df["next_close"].values[i])

X = np.array(features)
y_true = np.array(labels)
y_pred = model.predict(X)

# === 5. 計算策略報酬 ===
result = pd.DataFrame({
    "date": dates,
    "predict": y_pred,
    "real": y_true,
    "close": closes,
    "next_close": next_closes
})

result["return"] = (result["next_close"] - result["close"]) / result["close"]
result["strategy_return"] = result["return"] * result["predict"]

# === 6. 匯出預測結果 ===
result.to_csv("0050_predictions.csv", index=False)

# === 7. 顯示績效分析 ===
total_trades = len(result)
win_rate = (result[result["strategy_return"] > 0].shape[0] / total_trades) * 100
avg_return = result["strategy_return"].mean() * 100
cumulative_return = result["strategy_return"].sum() * 100
max_drawdown = result["strategy_return"].cumsum().min() * 100

print("📊 預測績效報告（0050_future）:")
print(f"➡️ 交易次數：{total_trades}")
print(f"✅ 勝率：{win_rate:.2f}%")
print(f"💰 平均單次報酬：{avg_return:.2f}%")
print(f"📈 累積報酬：{cumulative_return:.2f}%")
print(f"⚠️ 最大連續虧損：{max_drawdown:.2f}%")
print("✅ 已匯出預測結果至：0050_predictions.csv")



# === 8. 產生交易點位 json 檔（前端用） ===
trades = []

for i in range(len(result)):
    if result.iloc[i]["predict"] == 1:
        buy_date = result.iloc[i]["date"].strftime("%Y-%m-%d")
        sell_date = result.iloc[i + 1]["date"].strftime("%Y-%m-%d") if i + 1 < len(result) else None
        trades.append({
            "buy_date": buy_date,
            "sell_date": sell_date,
            "buy_price": round(result.iloc[i]["close"], 2),
            "sell_price": round(result.iloc[i]["next_close"], 2),
            "return": round(result.iloc[i]["strategy_return"] * 100, 2)
        })

with open("0050_trades.json", "w", encoding="utf-8") as f:
    json.dump(trades, f, indent=2, ensure_ascii=False)

print("✅ 已匯出交易點位至 0050_trades.json，供前端視覺化使用")
