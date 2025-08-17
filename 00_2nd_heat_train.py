### 2次加熱のみの
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 製品情報
products = ['7015', '7016', '7017', '9999']
first_frequencies = [2500, 5000, 7500, 10000]
first_currents = [2000, 4000, 6000, 8000, 10000]
second_frequencies = [2500, 5000, 7500]
second_currents = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
distances = {'7015':14.1, '7016':10.85, '7017':8.3, '9999':4.85}
weights = {'7015':227, '7016':296, '7017':316, '9999':385}
widths = {'7015':1405, '7016':15.65, '7017':15.65, '9999':15.65}
heights = {'7015':20.2, '7016':22.2, '7017':22.2, '9999':22.2}

sequence_data = []
target_data = []

# データ構築
for product in products:
  for freq_1 in first_frequencies:
    for cur_1 in first_currents:
      for freq_2 in second_frequencies:
        for cur_2 in second_currents:
          filename = f'{product}_1st_{freq_1}Hz_{cur_1}A_2nd_{freq_2}Hz_{cur_2}A.csv'
          if not os.path.exists(filename):
            continue
          
          df = pd.read_csv(filename)
          initial_row = df[df['time'] == 0]
          initial_max_temp = initial_row[[f'point_{j}' for j in range(1, 13)]].max().max()
          initial_min_temp = initial_row[[f'point_{j}' for j in range(1, 13)]].min().min()
          
          sequence = []
          for _, row in df.iterrows():
            elaspsed_time = row['time']
            max_temp = row[[f'point_{j}' for j in range(1, 13)]].max()
            min_temp = row[[f'point_{j}' for j in range(1, 13)]].min()
            features = [elaspsed_time, max_temp, min_temp, initial_max_temp, initial_min_temp, freq_2, weights[product], distances[product], widths[product], heights[product]]
            sequence.append(features)
          
          sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
          target_tensor = torch.tensor([cur_2], dtype=torch.float32)
          
          sequence_data.append(sequence_tensor)
          target_data.append(target_tensor)

# 特徴表の標準化
scaler = StandardScaler()
all_sequences = torch.cat(sequence_data, dim=0)
scaler.fit(all_sequences)
sequence_data_scaled = [torch.tensor(scaler.transform(seq), dtype=torch.float32) for seq in sequence_data]

# ターゲットの標準化
target_scaler = StandardScaler()
all_targets = torch.cat(target_data).numpy().reshape(-1, 1)
target_scaler.fit(all_targets)
scaled_targets = target_scaler.transform(all_targets)
target_data_scaled = [torch.tensor([val[0]], dtype=torch.float32) for val in scaled_targets]

# データセットの作成
X = torch.nn.utils.rnn.pad_sequence(sequence_data_scaled, batch_first=True)
y = torch.stack(target_data_scaled).squeeze(-1)
dataset = TensorDataset(X, y)
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# モデル定義
class BiLSTMWithAttention(nn.Model):
  def __init__(self, input_size, hidden_size=64):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
    
    self.attention = nn.Sequential(
      nn.Linear(hidden_size * 2, hidden_size),
      nn.Tanh(),
      nn.Linear(hidden_size, 1)
    )
    
    self.fc = nn.Sequential(
      nn.Linear(hidden_size * 2, 1),
      nn.Tanh()
    )
  
  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    atten_scores = self.attention(lstm_out)
    atten_weights = torch.softmax(atten_scores, dim=1)
    context = torch.sum(atten_weights*lstm_out, dim=1)
    output = self.fc(context).squeeze(-1)
    return output

model = BiLSTMWithAttention(input_size=X.shape[2])
criterion = nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
train_losses, val_losses = [], []
for epoch in range(50):
  model.train()
  train_loss = 0
  for xb, yb in train_loader:
    optimizer.zero_grad()
    pred = model(xb)
    loss = criterion(pred, yb)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
  train_loss /= len(train_loader)
  train_losses.append(train_loss)
  
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for xb, yb in val_loader:
      pred = model(xb)
      val_loss += criterion(pred, yb).item()
  val_loss /= len(val_loader)
  val_losses.append(val_loss)
  
  print(f"Epoch {epoch+1}:Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# モデルの保存
torch.save(model.state_dict(), "lstm_model_scaled.pth")
np.save("feature_scaler_mean.npy", scaler.mean_)
np.save("feature_scaler_std.npy", scaler.scale_)
np.save("target_scaler_mean.npy", target_scaler.mean_)
np.save("target_scaler_std.npy", target_scaler.scale_)

# 損失グラフ
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()