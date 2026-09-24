import pandas as pd
import torch
import torch.nn as nn

# --- Step 1: Recreate the exact same network architecture ---
class RuntimePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# --- Step 2: Load the saved weights into that architecture ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RuntimePredictor().to(device)
model.load_state_dict(torch.load("runtime_model.pth", map_location=device))
model.eval()

# --- Step 3: Load the first 10 rows from the original dataset ---
df = pd.read_csv("training_data.csv")
sample = df.head(10)

X_sample = sample[['ncpus', 'mem_gb', 'submit_hour', 'queue_type']].values
y_actual = sample['actual_runtime'].values

X_sample_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)

# --- Step 4: Run predictions ---
with torch.no_grad():
    y_predicted = model(X_sample_tensor).cpu().numpy().flatten()

# --- Step 5: Display comparison table ---
comparison = pd.DataFrame({
    "ncpus": sample['ncpus'].values,
    "mem_gb": sample['mem_gb'].values,
    "submit_hour": sample['submit_hour'].values,
    "queue_type": sample['queue_type'].values,
    "actual_runtime": y_actual,
    "predicted_runtime": y_predicted,
    "abs_error": abs(y_actual - y_predicted)
})

print("=== Comparison: First 10 Rows ===")
print(comparison.to_string(index=False))

# --- Step 6: Overall accuracy metric across these 10 rows ---
mae = comparison['abs_error'].mean()
mean_actual = y_actual.mean()
accuracy_pct = 100 * (1 - (mae / mean_actual))

print(f"\nMean Absolute Error (first 10 rows): {mae:.2f}")
print(f"Approximate accuracy: {accuracy_pct:.2f}%")