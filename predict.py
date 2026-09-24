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

# --- Step 3: Load the FULL dataset ---
df = pd.read_csv("training_data.csv")

X_all = df[['ncpus', 'mem_gb', 'submit_hour', 'queue_type']].values
y_actual = df['actual_runtime'].values

X_all_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)

# --- Step 4: Run predictions across ALL rows ---
with torch.no_grad():
    y_predicted = model(X_all_tensor).cpu().numpy().flatten()

# --- Step 5: Build the full comparison table ---
comparison = pd.DataFrame({
    "ncpus": df['ncpus'].values,
    "mem_gb": df['mem_gb'].values,
    "submit_hour": df['submit_hour'].values,
    "queue_type": df['queue_type'].values,
    "actual_runtime": y_actual,
    "predicted_runtime": y_predicted,
    "abs_error": abs(y_actual - y_predicted)
})

# --- Step 6: Display only a readable preview ---
print("=== First 10 rows (preview) ===")
print(comparison.head(10).to_string(index=False))

# --- Step 7: Save the FULL comparison to a file, for later inspection if needed ---
comparison.to_csv("full_predictions.csv", index=False)
print(f"\nFull comparison for all {len(df)} rows saved to full_predictions.csv")

# --- Step 8: Accuracy metrics calculated across the ENTIRE dataset ---
mae = comparison['abs_error'].mean()
mean_actual = y_actual.mean()
accuracy_pct = 100 * (1 - (mae / mean_actual))

print(f"\nMean Absolute Error (all {len(df)} rows): {mae:.2f}")
print(f"Approximate accuracy: {accuracy_pct:.2f}%")