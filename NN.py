import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wall_models_dir = "/home/onkar/neko/src/wall_models"
os.makedirs(wall_models_dir, exist_ok=True)
yaml_path = os.path.join(wall_models_dir, "model_config.yaml")
pt_path = os.path.join(wall_models_dir, "model_scripted.pt")
scaler_X_mean_path = os.path.join(wall_models_dir, "scaler_X_mean.npy")
scaler_X_scale_path = os.path.join(wall_models_dir, "scaler_X_scale.npy")
scaler_Y_mean_path = os.path.join(wall_models_dir, "scaler_Y_mean.npy")
scaler_Y_scale_path = os.path.join(wall_models_dir, "scaler_Y_scale.npy")
scaler_X_mean_dat = os.path.join(wall_models_dir, "scaler_X_mean.dat")
scaler_X_scale_dat = os.path.join(wall_models_dir, "scaler_X_scale.dat")
scaler_Y_mean_dat = os.path.join(wall_models_dir, "scaler_Y_mean.dat")
scaler_Y_scale_dat = os.path.join(wall_models_dir, "scaler_Y_scale.dat")
plot_path = os.path.join(wall_models_dir, "training_validation_testing_and_apriori_plots.png")
output_path = '/home/onkar/neko/examples/predictionsonnewsimulationdata.xlsx'

# --- Load and preprocess ---
path = '/home/onkar/neko/examples/WRLES data.xlsx'
original_input_cols = ['U', 'V', 'W', 'Pressure', 'nu']
output_cols = ['Shear Stress']
all_sheets = pd.read_excel(path, sheet_name=None)
X_train, Y_train = [], []
X_apriori, Y_apriori, t_apriori = [], [], []
max_nodes = 4
feat_per_node = len(original_input_cols)
save_dict = {}

for sheet_name, df in all_sheets.items():
    df = df.dropna(subset=original_input_cols + output_cols)
    df = df.copy()
    df['log_Re'] = np.log(df['Re'])
    grouped = df.groupby('Time', group_keys=False)
    for time_val, group in grouped:
        group_filtered = group[((group['Node'] >= -1.0) & (group['Node'] <= -0.9))].sort_values(by='Node')
        if len(group_filtered) == 0:
            continue
        key = f"{sheet_name}_Time{time_val}"
        save_dict[key] = group_filtered
        if len(group_filtered) >= max_nodes:
            window = group_filtered.iloc[0:max_nodes]
            if np.all(np.diff(window.index) == 1):
                features_per_node = ['U', 'V', 'W', 'Pressure']
                features_nodes = window[features_per_node].values.flatten()
                nu_value = window['nu'].iloc[0]
                features_concat = np.concatenate([features_nodes, [nu_value]])
                target_avg = window[output_cols].mean().values
                target_avg = np.clip(target_avg, a_min=1e-8, a_max=None)
                if sheet_name == "4000":
                    X_apriori.append(features_concat)
                    Y_apriori.append(target_avg)
                    t_apriori.append(time_val)
                else:
                    X_train.append(features_concat)
                    Y_train.append(target_avg)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
X_apriori = np.array(X_apriori, dtype=np.float32)
Y_apriori = np.array(Y_apriori, dtype=np.float32)

# --- NN model ---
class NeuralNet(nn.Module):
    def __init__(self,dropout_rate=0.05):
        super().__init__()
        self.fc0 = nn.Linear(17, 15)
        self.fc1 = nn.Linear(15, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 15)
        self.fc4 = nn.Linear(15, 15)
        self.fc5 = nn.Linear(15, 15)
        self.fc6 = nn.Linear(15, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.dropout(self.activation(self.fc0(x)))
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))
        x = self.dropout(self.activation(self.fc5(x)))
        x = self.fc6(x)
        return x

# --- Initial Training (with validation and early stopping with min_delta) ---
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

# Split for validation
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train_scaled, Y_train_scaled, test_size=0.2, random_state=42)
X_tr_tensor = torch.FloatTensor(X_tr).to(device)
Y_tr_tensor = torch.FloatTensor(Y_tr).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
Y_val_tensor = torch.FloatTensor(Y_val).to(device)

train_dataset = TensorDataset(X_tr_tensor, Y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Prepare test data tensors for loss calculation
X_test_scaled = scaler_X.transform(X_apriori)
Y_test_scaled = scaler_Y.transform(Y_apriori)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
Y_test_tensor = torch.FloatTensor(Y_test_scaled).to(device)

model = NeuralNet(dropout_rate=0.05).float().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

epochs = 500

train_losses, val_losses, test_losses = [], [], []

min_delta = 1e-5               
patience = 5                  
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor)
        val_losses.append(val_loss.item())

        # Test loss
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, Y_test_tensor)
        test_losses.append(test_loss.item())

    scheduler.step(running_loss)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.6f}, Val: {val_loss.item():.6f}, Test: {test_loss.item():.6f}")

    if best_val_loss - val_loss.item() > min_delta:
        best_val_loss = val_loss.item()
        patience_counter = 0  # reset counter if improvement
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}: no improvement beyond {min_delta} for {patience} epochs.")
        break

# --- A Priori Prediction ---
model.eval()
with torch.no_grad():
    Y_apriori_scaled = model(X_test_tensor)
    Y_apriori_pred = scaler_Y.inverse_transform(Y_apriori_scaled.cpu().numpy())

# --- Plot Results (1x4 subplots) ---
fig, axs = plt.subplots(1, 4, figsize=(24, 6))
axs[0].plot(t_apriori, Y_apriori[:, 0], label='True Shear Stress', linewidth=2)
axs[0].plot(t_apriori, Y_apriori_pred[:, 0], '--', label='Predicted Shear Stress', color='red', linewidth=2)
axs[0].set_title('A Priori Shear Stress')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Shear Stress')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(1, len(train_losses) + 1), train_losses, color='blue')
#axs[1].set_xscale('log')
axs[1].set_title('Training Loss')
axs[1].set_xlabel('Epoch (log)')
axs[1].set_ylabel('MSE Loss')
axs[1].grid(True)

axs[2].plot(range(1, len(val_losses) + 1), val_losses, color='orange')
#axs[2].set_xscale('log')
axs[2].set_title('Validation Loss')
axs[2].set_xlabel('Epoch (log)')
axs[2].set_ylabel('MSE Loss')
axs[2].grid(True)

axs[3].plot(range(1, len(test_losses) + 1), test_losses, color='green')
#axs[3].set_xscale('log')
axs[3].set_title('Testing Loss')
axs[3].set_xlabel('Epoch (log)')
axs[3].set_ylabel('MSE Loss')
axs[3].grid(True)

plt.tight_layout()
plt.savefig(plot_path)

# --- Save predictions to Excel ---
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_apriori = pd.DataFrame({
        'Time': t_apriori,
        'True_Shear_Stress': Y_apriori[:, 0],
        'Predicted_Shear_Stress': Y_apriori_pred[:, 0]
    })
    df_apriori.to_excel(writer, sheet_name='A Priori', index=False)
print(f"Predictions saved to: {output_path}")

# --- Save normalization statistics ---
np.save(scaler_X_mean_path, scaler_X.min_.astype(np.float32))
np.save(scaler_X_scale_path, scaler_X.scale_.astype(np.float32))
np.save(scaler_Y_mean_path, scaler_Y.data_min_.astype(np.float32))
np.save(scaler_Y_scale_path, scaler_Y.data_max_.astype(np.float32))

def npy_to_dat(npy_file, dat_file):
    data = np.load(npy_file)
    np.savetxt(dat_file, data, fmt='%.18e')

npy_to_dat(scaler_X_mean_path, scaler_X_mean_dat)
npy_to_dat(scaler_X_scale_path, scaler_X_scale_dat)
npy_to_dat(scaler_Y_mean_path, scaler_Y_mean_dat)
npy_to_dat(scaler_Y_scale_path, scaler_Y_scale_dat)
print("Scaler .npy files converted to .dat files.")

# --- TorchFort/Scripted Model Saving ---
class RenamedMLP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.w0 = model.fc0.weight
        self.b0 = model.fc0.bias
        self.w1 = model.fc1.weight
        self.b1 = model.fc1.bias
        self.w2 = model.fc2.weight
        self.b2 = model.fc2.bias
        self.w3 = model.fc3.weight
        self.b3 = model.fc3.bias
        self.w4 = model.fc4.weight
        self.b4 = model.fc4.bias
        self.w5 = model.fc5.weight
        self.b5 = model.fc5.bias
        self.w6 = model.fc6.weight
        self.b6 = model.fc6.bias
        self.activation = model.activation
    def forward(self, x):
        x = self.activation(nn.functional.linear(x, self.w0, self.b0))
        x = self.activation(nn.functional.linear(x, self.w1, self.b1))
        x = self.activation(nn.functional.linear(x, self.w2, self.b2))
        x = self.activation(nn.functional.linear(x, self.w3, self.b3))
        x = self.activation(nn.functional.linear(x, self.w4, self.b4))
        x = self.activation(nn.functional.linear(x, self.w5, self.b5))
        x = nn.functional.linear(x, self.w6, self.b6)
        return x

model_cpu = model.cpu()
renamed_model = RenamedMLP(model_cpu).eval()
scripted_model = torch.jit.script(renamed_model)
scripted_model.save(pt_path)
print(f"TorchScript model saved as {pt_path}")

def export_torchfort_yaml_config(filename=yaml_path, torchscript_file=pt_path):
    config = {
        "model": {
            "type": "torchscript",
            "parameters": {
                "filename": torchscript_file
            }
        }
    }
    with open(filename, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"TorchFort YAML model config saved to {filename}")

if __name__ == "__main__":
    export_torchfort_yaml_config()
