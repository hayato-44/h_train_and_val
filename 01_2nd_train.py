# -*- coding: utf-8 -*-
"""
train_seq2seq_current.py
- 2次加熱（6秒, 0.05s刻み想定）の「推奨電流の時系列」を出力する BiLSTM の学習スクリプト
- 入力: 時刻特徴（tのフーリエ展開） + 固定条件（距離,重量,寸法,初期温度帯,狙い温度帯,2次周波数）
- 教師: 電流の時系列（CSVに current 列が無ければ、ファイル名の 2nd 電流で全時刻を埋める）
- 出力: checkpoint (モデル+スケーラー+メタ情報)
"""
import os, re, glob
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler

# ===== ユーザー環境設定 =====
CSV_DIR   = "./"    # CSVの置き場所
DT        = 0.05    # サンプリング間隔 [s]
DURATION  = 6.0     # 2次加熱の長さ [s]
FOURIER_H = 2       # 時刻特徴のフーリエ次数（1〜2で十分）

# 製品ごとの定数（必要に応じて実値に更新）
distances = {'7015':14.1,  '7016':10.85, '7017':8.3,  '9999':4.85}
weights   = {'7015':227,   '7016':296,   '7017':316,  '9999':385}
widths    = {'7015':14.05, '7016':15.65, '7017':15.65,'9999':15.65} 
heights   = {'7015':20.2,  '7016':22.2,  '7017':22.2, '9999':22.2}

# 狙い温度帯（学習入力用の固定条件。運用時は推論側から与える）
DEFAULT_TARGET_MAX = 950.0
DEFAULT_TARGET_MIN = 920.0

# ===== ファイル名パターン（1次条件は読み捨て。2次は使用）=====
FNAME_RE = re.compile(
    r'(?P<prod>\d+)_1st_(?P<f1>\d+)Hz_(?P<c1>\d+)A_2nd_(?P<f2>\d+)Hz_(?P<c2>\d+)A\.csv$'
)
def parse_filename(path:str):
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    return {"product": d["prod"], "f1": int(d["f1"]), "c1": float(d["c1"]),
            "f2": int(d["f2"]),  "c2": float(d["c2"])}

# ===== 特徴量ユーティリティ =====
def time_features(t: np.ndarray, total: float, H:int=2) -> np.ndarray:
    """
    t: (T,)  ->  return: (T, 1+2H)  [t, sin(2πht/total), cos(2πht/total)] for h=1..H
    """
    feats = [t.reshape(-1,1)]
    for h in range(1, H+1):
        feats.append(np.sin(2*np.pi*h*t/total).reshape(-1,1))
        feats.append(np.cos(2*np.pi*h*t/total).reshape(-1,1))
    return np.concatenate(feats, axis=1).astype(np.float32)

# ===== Dataset =====
class HeatingDataset(Dataset):
    def __init__(self, csv_dir:str, use_constant_labels_if_missing=True,
                 target_max=DEFAULT_TARGET_MAX, target_min=DEFAULT_TARGET_MIN):
        self.paths = [p for p in glob.glob(os.path.join(csv_dir, "*.csv"))
                      if FNAME_RE.match(os.path.basename(p))]
        if not self.paths:
            raise FileNotFoundError("CSVが見つかりません。命名規約を確認してください。")
        self.use_constant = use_constant_labels_if_missing
        self.tgt_max = target_max
        self.tgt_min = target_min

        # まず素データでfit用に収集
        dyn_list, stat_list, y_list = [], [], []
        for p in self.paths:
            dyn, stat, y = self._build_one(p)
            dyn_list.append(dyn)    # (T, D_dyn)
            stat_list.append(stat)  # (D_stat,)
            y_list.append(y)        # (T,)

        dyn_all  = np.concatenate(dyn_list, axis=0)     # (sum_T, D_dyn)
        stat_all = np.stack(stat_list, axis=0)          # (N, D_stat)
        y_all    = np.concatenate(y_list, axis=0).reshape(-1,1)

        self.dyn_scaler  = StandardScaler().fit(dyn_all)
        self.stat_scaler = StandardScaler().fit(stat_all)
        self.y_scaler    = StandardScaler().fit(y_all)

        # スケール後をメモリ保持
        self.items = []
        for p in self.paths:
            dyn, stat, y = self._build_one(p)
            dyn  = self.dyn_scaler.transform(dyn).astype(np.float32)
            stat = self.stat_scaler.transform(stat.reshape(1,-1)).astype(np.float32)[0]
            y    = self.y_scaler.transform(y.reshape(-1,1)).astype(np.float32).reshape(-1)
            self.items.append((
                torch.tensor(dyn,  dtype=torch.float32),
                torch.tensor(stat, dtype=torch.float32),
                torch.tensor(y,    dtype=torch.float32)
            ))

        self.dyn_in  = dyn_list[0].shape[-1]
        self.stat_in = stat_list[0].shape[-1]

    def _build_one(self, path:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        meta = parse_filename(path)
        if meta is None:
            raise ValueError(f"命名規約外のファイル: {path}")

        df = pd.read_csv(path)
        t = df["time"].values.astype(np.float32)  # 可変長でもOK
        # 初期温度（空冷直後の時刻=0行から）
        point_cols = [c for c in df.columns if c.startswith("point_")]
        init_row   = df.iloc[0]
        init_max   = float(init_row[point_cols].max())
        init_min   = float(init_row[point_cols].min())

        prod = meta["product"]
        # 固定特徴（この順序は checkpoint に保存され、推論時に再使用）
        stat = np.array([
            distances[prod], weights[prod], widths[prod], heights[prod],  # 幾何/重量
            init_max, init_min,                                           # 初期温度帯（空冷直後）
            self.tgt_max, self.tgt_min,                                   # 狙い温度帯
            meta["f2"],                                                   # 2次周波数のみ使用
        ], dtype=np.float32)

        dyn = time_features(t, total=DURATION, H=FOURIER_H)               # (T, D_dyn)

        # 教師電流：CSVに current 列があれば使用、無ければ c2 を全時刻に敷く
        if "current" in df.columns:
            y = df["current"].values.astype(np.float32)
        else:
            if not self.use_constant:
                raise ValueError("current 列がありません。可変波形を教師化するには current 列が必要です。")
            y = np.full(len(t), fill_value=meta["c2"], dtype=np.float32)

        return dyn, stat, y

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def collate_fn(batch):
    seqs_dyn, stats, ys = zip(*batch)  # list of (T,Dd), (Ds,), (T,)
    lengths = torch.tensor([len(s) for s in seqs_dyn], dtype=torch.long)
    pad_dyn = pad_sequence(seqs_dyn, batch_first=True)  # (B, T_max, Dd)
    pad_y   = pad_sequence(ys,       batch_first=True)  # (B, T_max)
    stats   = torch.stack(stats, dim=0)                 # (B, Ds)
    mask = torch.arange(pad_dyn.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return pad_dyn, stats, pad_y, lengths, mask.bool()

# ===== モデル =====
class Seq2SeqCurrent(nn.Module):
    def __init__(self, dyn_in:int, stat_in:int,
                 stat_embed:int=64, hidden:int=128, num_layers:int=2,
                 bidirectional:bool=False, dropout:float=0.1):
        super().__init__()
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_in, 128), nn.ReLU(),
            nn.Linear(128, stat_embed), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=dyn_in + stat_embed,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x_dyn, x_stat, lengths):
        B, T, _ = x_dyn.shape
        s = self.stat_mlp(x_stat)                     # (B, stat_embed)
        s_rep = s.unsqueeze(1).expand(B, T, s.size(-1))
        x = torch.cat([x_dyn, s_rep], dim=-1)         # (B, T, dyn+embed)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        yhat = self.head(out).squeeze(-1)             # (B, T)
        return yhat

def masked_mse(pred, target, mask):
    diff = (pred - target)**2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff.sum() / denom

def main():
    torch.manual_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = HeatingDataset(CSV_DIR, use_constant_labels_if_missing=True)
    # データ数に応じて分割
    val_ratio = 0.2 if len(ds) >= 5 else 0.0
    if val_ratio > 0:
        val_size   = int(len(ds) * val_ratio)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = ds, None

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=collate_fn) if val_ds else None

    model = Seq2SeqCurrent(dyn_in=ds.dyn_in, stat_in=ds.stat_in).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)

    best_val = float('inf')
    EPOCHS = 50
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss = 0.0
        for x_dyn, x_stat, y, lengths, mask in train_dl:
            x_dyn, x_stat, y, mask = x_dyn.to(DEVICE), x_stat.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            pred = model(x_dyn, x_stat, lengths)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        if val_dl:
            model.eval(); va_loss = 0.0
            with torch.no_grad():
                for x_dyn, x_stat, y, lengths, mask in val_dl:
                    x_dyn, x_stat, y, mask = x_dyn.to(DEVICE), x_stat.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                    pred = model(x_dyn, x_stat, lengths)
                    va_loss += masked_mse(pred, y, mask).item()
            va_loss /= len(val_dl)
            sch.step(va_loss)
            print(f"Epoch {ep:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")
            if va_loss < best_val:
                best_val = va_loss
                torch.save({
                    "model_state": model.state_dict(),
                    # スケーラー
                    "dyn_mean":  ds.dyn_scaler.mean_,  "dyn_scale":  ds.dyn_scaler.scale_,
                    "stat_mean": ds.stat_scaler.mean_, "stat_scale": ds.stat_scaler.scale_,
                    "y_mean":    ds.y_scaler.mean_,    "y_scale":    ds.y_scaler.scale_,
                    # 入出力次元・ハイパラ
                    "dyn_in": ds.dyn_in, "stat_in": ds.stat_in,
                    "stat_embed": 64, "hidden":128, "num_layers":2, "bidirectional":False, "dropout":0.1,
                    # 時間関連メタ
                    "duration": DURATION, "dt": DT, "fourier_h": FOURIER_H,
                    # 固定特徴の順序
                    "stat_order": [
                        "distance","weight","width","height",
                        "initial_max","initial_min",
                        "target_max","target_min",
                        "second_freq"
                    ],
                }, "seq2seq_current.pth")
        else:
            # val無しでも都度保存
            torch.save({
                "model_state": model.state_dict(),
                "dyn_mean":  ds.dyn_scaler.mean_,  "dyn_scale":  ds.dyn_scaler.scale_,
                "stat_mean": ds.stat_scaler.mean_, "stat_scale": ds.stat_scaler.scale_,
                "y_mean":    ds.y_scaler.mean_,    "y_scale":    ds.y_scaler.scale_,
                "dyn_in": ds.dyn_in, "stat_in": ds.stat_in,
                "stat_embed": 64, "hidden":128, "num_layers":2, "bidirectional":False, "dropout":0.1,
                "duration": DURATION, "dt": DT, "fourier_h": FOURIER_H,
                "stat_order": [
                    "distance","weight","width","height",
                    "initial_max","initial_min",
                    "target_max","target_min",
                    "second_freq"
                ],
            }, "seq2seq_current.pth")
            print(f"Epoch {ep:02d} | Train {tr_loss:.4f}")

    print("Training done.")

if __name__ == "__main__":
    main()