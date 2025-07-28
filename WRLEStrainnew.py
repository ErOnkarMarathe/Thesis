import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ===============================
# Setup
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Data Paths and Columns
# ===============================

path = r"\\wsl.localhost\Ubuntu\home\onkar\neko\examples\WRLES Data.xlsx"
wmles_path = r"\\wsl.localhost\Ubuntu\home\onkar\neko\examples\WMLES\fluid_stats0.xlsx"
output_path = r"\\wsl.localhost\Ubuntu\home\onkar\neko\examples\WMLES_with_predictions.xlsx"

input_cols = ['Time', 'Node','Pressure', 'Normalized U', 'Normalized V', 'Normalized W','Pol Order', 'No. Of Ele.']
#input_cols = ['Time', 'Node', 'Pressure', 'U', 'V', 'W', 'Pol Order', 'No. Of Ele.']
#output_col = ['Wall Shear stress due to viscosity (tau xy)', 'Wall Shear stress due to viscosity (tau zy)']
output_col = ['Normalized Wall Shear stress due to viscosity (tau xy)', 'Normalized Wall Shear stress due to viscosity (tau zy)']

# ===============================
# Load WRLES Data (all sheets)
# ===============================

all_sheets = pd.read_excel(path, sheet_name=None)

print("Available sheets in WRLES Excel file:")
print(all_sheets.keys())  # Check actual sheet names to avoid KeyError

# ===============================
# Prepare training data from WRLES
# ===============================

X_all, Y_all = [], []

for sheet_name, df in all_sheets.items():
    df = df.dropna(subset=input_cols + output_col)
    # Fix: keep 'Time' column by using group_keys=False and no reset_index(drop=True)
    df_filtered = df.groupby('Time', group_keys=False).apply(lambda g: g.sort_values(by='Node').head(4))

    if len(df_filtered) < 4:
        continue

    X = df_filtered[input_cols].values
    Y = df_filtered[output_col].values
    X_all.append(X)
    Y_all.append(Y)

X_all = np.vstack(X_all)
Y_all = np.vstack(Y_all)
print(f"Total samples: {len(X_all)}")
print(f"Variance of target: {Y_all.var()}")

# ===============================
# Normalize Data
# ===============================

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_all)
Y_scaled = scaler_Y.fit_transform(Y_all)

X_tensor = torch.FloatTensor(X_scaled).to(device)
Y_tensor = torch.FloatTensor(Y_scaled).to(device)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# ===============================
# MLP Model Definition
# ===============================

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

model = NeuralNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# ===============================
# Training Loop
# ===============================

epochs = 1000
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, Y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.6f}")

# ===============================
# Prediction on WMLES Data (First 4 Nodes per Time)
# ===============================

wmles_df = pd.read_excel(wmles_path, sheet_name='fluid_stats0')
wmles_df = wmles_df.dropna(subset=input_cols)

# Fix: keep 'Time' column here as well
wmles_filtered = wmles_df.groupby('Time', group_keys=False).apply(lambda g: g.sort_values(by='Node').head(4))

X_new = wmles_filtered[input_cols].values
X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.FloatTensor(X_new_scaled).to(device)

model.eval()
with torch.no_grad():
    Y_pred_scaled = model(X_new_tensor)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.cpu().numpy())

wmles_filtered['Predicted tau xy'] = Y_pred[:, 0]
wmles_filtered['Predicted tau zy'] = Y_pred[:, 1]

wmles_filtered.to_excel(output_path, index=False)
print(f"Saved predictions to: {output_path}")

# ===============================
# Plotting: WRLES (Re 2300, 2800, 4000, 10000) and WMLES predicted tau_xy
# ===============================

# Adjust these sheet names to match your actual Excel sheet names!
re_sheets = {
    2300: '2300',
    2800: '2800',
    4000: '4000',
    10000: '10000'
}

plt.figure(figsize=(12, 6))

for re, sheet_name in re_sheets.items():
    if sheet_name not in all_sheets:
        print(f"Warning: Sheet '{sheet_name}' not found in WRLES data. Skipping Re={re}.")
        continue

    df = all_sheets[sheet_name].dropna(subset=input_cols + output_col)
    df_filtered = df.groupby('Time', group_keys=False).apply(lambda g: g.sort_values(by='Node').head(4))

    # Average tau_xy over first 4 nodes per time for smoother plot
    avg_tau = df_filtered.groupby('Time')['Wall Shear stress due to viscosity (tau xy)'].mean()
    plt.plot(avg_tau.index, avg_tau.values, label=f'WRLES Re={re}')

# WMLES predicted average tau_xy over first 4 nodes per time
avg_pred_tau = wmles_filtered.groupby('Time')['Predicted tau xy'].mean()
plt.plot(avg_pred_tau.index, avg_pred_tau.values, '--', label='WMLES Predicted', color='black')

plt.title(r'$\tau_{xy}$ for WRLES (Re=2300, 2800, 4000, 10000) and WMLES Predicted')
plt.xlabel('Time')
plt.ylabel(r'$\tau_{xy}$')
plt.legend()
plt.tight_layout()
plt.show()
