import json
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# === 1. 載入 0050 資料 ===
with open("../output/0050_candles.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["data"])
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.sort_index()

# === 2. 技術指標 ===
df["ma5"] = SMAIndicator(df["close"], window=5).sma_indicator()
df["ma10"] = SMAIndicator(df["close"], window=10).sma_indicator()
df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
df = df.dropna()

# === 3. 建立漲跌標籤（明日收盤 > 今日收盤 => 1）===
df["next_close"] = df["close"].shift(-1)
df["label"] = (df["next_close"] > df["close"]).astype(int)
df = df.dropna()

# === 4. 滑動視窗特徵產生 ===
window_size = 10
features, labels, dates = [], [], []

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

X = np.array(features)
y = np.array(labels)

print(f"✅ 0050 訓練樣本數: {len(X)}")

# === 5. 模型訓練 ===
model = XGBClassifier(eval_metric="logloss")
model.fit(X, y)

# === 6. 預測與評估 ===
y_pred = model.predict(X)
print("📈 訓練集準確率:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# === 7. 儲存模型 ===
with open("xgb_model_0050.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ 模型已訓練並儲存為 xgb_model_0050.pkl")
