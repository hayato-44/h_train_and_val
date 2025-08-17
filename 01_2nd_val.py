# -*- coding: utf-8 -*-
"""
infer_seq2seq_current.py
- 学習済みcheckpointから、時間ごとの推奨電流[A]を推論
- 指定区間 [start_time, end_time]、任意刻み step_dt で推論可能
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ===== モデル（学習側と同定義）=====
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
            dropout=dropout if num_layers>1 else 0.0
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x_dyn, x_stat, lengths):
        B, T, _ = x_dyn.shape
        s = self.stat_mlp(x_stat)
        s_rep = s.unsqueeze(1).expand(B, T, s.size(-1))
        x = torch.cat([x_dyn, s_rep], dim=-1)
        out, _ = self.lstm(x)           # 推論は一定長なのでpack不要
        yhat = self.head(out).squeeze(-1)
        return yhat

# ===== 時刻特徴 =====
def time_features(t: np.ndarray, total: float, H:int=2) -> np.ndarray:
    # total には「学習時の総時間（=6.0秒）」を入れるとスケーラーと整合が取りやすい
    feats = [t.reshape(-1,1)]
    for h in range(1, H+1):
        feats.append(np.sin(2*np.pi*h*t/total).reshape(-1,1))
        feats.append(np.cos(2*np.pi*h*t/total).reshape(-1,1))
    return np.concatenate(feats, axis=1).astype(np.float32)

def load_checkpoint(ckpt_path="seq2seq_current.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # スケーラー復元
    dyn_scaler = StandardScaler()
    dyn_scaler.mean_  = ckpt["dyn_mean"];  dyn_scaler.scale_ = ckpt["dyn_scale"]

    stat_scaler = StandardScaler()
    stat_scaler.mean_ = ckpt["stat_mean"]; stat_scaler.scale_ = ckpt["stat_scale"]

    y_scaler = StandardScaler()
    y_scaler.mean_    = ckpt["y_mean"];    y_scaler.scale_   = ckpt["y_scale"]

    # モデル復元
    model = Seq2SeqCurrent(
        dyn_in=ckpt["dyn_in"], stat_in=ckpt["stat_in"],
        stat_embed=ckpt["stat_embed"], hidden=ckpt["hidden"],
        num_layers=ckpt["num_layers"], bidirectional=ckpt["bidirectional"],
        dropout=ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "duration":  ckpt["duration"],   # 学習時の総時間（例: 6.0）
        "dt":        ckpt["dt"],         # 学習時の刻み（例: 0.05）
        "fourier_h": ckpt["fourier_h"],
        "stat_order": ckpt["stat_order"],  # ["distance","weight","width","height","initial_max","initial_min","target_max","target_min","second_freq"]
    }
    return model, dyn_scaler, stat_scaler, y_scaler, meta, device

def predict_current_schedule(
    SPECIFIC_DISTANCE = 14.1,
    INITIAL_MAX_TEMP  = 901.0,
    INITIAL_MIN_TEMP  = 876.0,
    TARGET_MAX_TEMP   = 950.0,
    TARGET_MIN_TEMP   = 920.0,
    SPECIFIC_FREQUENCY= 8200,   # 2次周波数のみ
    SPECIFIC_WEIGHT   = 227.0,
    SPECIFIC_WIDTH    = 14.05,
    SPECIFIC_HEIGHT   = 20.2,
    # ここから区間/刻みを指定
    start_time        = 4.0,    # 推論区間の開始[s]
    end_time          = 6.0,    # 推論区間の終了[s]
    step_dt           = 0.1,    # 推論刻み[s]
    ckpt_path         = "seq2seq_current.pth"
):
    """
    指定区間 [start_time, end_time] を step_dt 刻みで推論します。
    フーリエ特徴の正規化は、学習時の総時間 meta["duration"] を使用します。
    """
    assert end_time > start_time, "end_time は start_time より大きい必要があります"
    assert step_dt > 0, "step_dt は正の値で指定してください"

    model, dyn_scaler, stat_scaler, y_scaler, meta, device = load_checkpoint(ckpt_path)

    train_total = meta["duration"]   # 学習時の総時間（例: 6.0）
    fourier_h   = meta["fourier_h"]

    # 時刻ベクトル（端点を含むように丸め込み）
    n_steps = int(round((end_time - start_time) / step_dt)) + 1
    t = start_time + np.arange(n_steps, dtype=np.float32) * step_dt

    # 時刻特徴は「学習時の総時間」を正規化の基準にして作る（スケーラー整合のため）
    dyn = time_features(t, total=train_total, H=fourier_h)
    dyn = dyn_scaler.transform(dyn).astype(np.float32)

    # 固定特徴（学習時と同順）
    stat_vec = np.array([
        SPECIFIC_DISTANCE, SPECIFIC_WEIGHT, SPECIFIC_WIDTH, SPECIFIC_HEIGHT,
        INITIAL_MAX_TEMP,  INITIAL_MIN_TEMP,
        TARGET_MAX_TEMP,   TARGET_MIN_TEMP,
        SPECIFIC_FREQUENCY,
    ], dtype=np.float32).reshape(1,-1)
    stat = stat_scaler.transform(stat_vec).astype(np.float32)

    x_dyn   = torch.tensor(dyn,  dtype=torch.float32).unsqueeze(0).to(device) # (1,T,D_dyn)
    x_stat  = torch.tensor(stat, dtype=torch.float32).to(device)              # (1,D_stat)
    lengths = torch.tensor([len(t)], dtype=torch.long)

    with torch.no_grad():
        y_scaled = model(x_dyn, x_stat, lengths)   # (1,T)
    y_scaled = y_scaled.squeeze(0).cpu().numpy()
    y = y_scaler.inverse_transform(y_scaled.reshape(-1,1)).reshape(-1)  # 実電流[A]

    return t, y

if __name__ == "__main__":
    # 例：4.0〜6.0s を 0.1s 刻みで推論
    t, I = predict_current_schedule(
        SPECIFIC_DISTANCE = 14.1,
        INITIAL_MAX_TEMP  = 901,
        INITIAL_MIN_TEMP  = 876,
        TARGET_MAX_TEMP   = 950,
        TARGET_MIN_TEMP   = 920,
        SPECIFIC_FREQUENCY= 8200,
        SPECIFIC_WEIGHT   = 227,
        SPECIFIC_WIDTH    = 14.05,
        SPECIFIC_HEIGHT   = 20.2,
        start_time        = 4.0,
        end_time          = 6.0,
        step_dt           = 0.1,
        ckpt_path         = "seq2seq_current.pth"
    )
    for i in range(len(t)):
        print(f"{t[i]:.2f}s, {I[i]:.1f} A")