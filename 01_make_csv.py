import numpy as np
import pandas as pd
import os

# 出力ディレクトリ
os.makedirs("sample_csv", exist_ok=True)

# 時刻ベクトル
dt = 0.05
T = int(6.0/dt) + 1  # 0〜6秒まで
time = np.arange(T) * dt

def make_csv(product="7015", f1=5000, c1=6000, f2=8200, c2=4000, with_current=True):
    # 温度: ダミーで "200℃ + 上昇分"
    base_temp = 200
    temps = []
    for i in range(12):
        # 各測定点は「上昇率を少しずらす」
        rise = (50 + i*2) * (time/6.0)  # 0→最大
        temps.append(base_temp + rise + np.random.normal(0, 1, size=len(time)))
    temps = np.array(temps).T  # (T, 12)

    # DataFrame化
    df = pd.DataFrame({"time": time})
    for j in range(12):
        df[f"point_{j+1}"] = temps[:, j]

    # current列をオプションで追加
    if with_current:
        # ダミー: 最初は c2-500A から徐々に c2 に近づく
        current = c2 - 500 + 500 * (time/6.0)
        df["current"] = current

    # ファイル名（命名規則に従う）
    fname = f"{product}_1st_{f1}Hz_{c1}A_2nd_{f2}Hz_{c2}A.csv"
    path = os.path.join("sample_csv", fname)
    df.to_csv(path, index=False)
    print("生成:", path)

# サンプルを複数作る
make_csv(product="7015", f1=5000, c1=6000, f2=8200, c2=4000, with_current=True)
make_csv(product="7016", f1=7500, c1=8000, f2=5000, c2=4500, with_current=False)
make_csv(product="7017", f1=10000,c1=2000, f2=7500, c2=5000, with_current=True)