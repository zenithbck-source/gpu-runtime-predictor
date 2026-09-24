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
df_training = pd.read_csv("training_data.csv")
X_all_training = df_training[['ncpus', 'mem_gb', 'submit_hour', 'queue_type']].values
y_actual_training = df_training['actual_runtime'].values

X_all_training_tensor = torch.tensor(X_all_training, dtype=torch.float32).to(device)

# --- Step 4: Run predictions across ALL rows ---
with torch.no_grad():
    y_predicted_training = model(X_all_training_tensor).cpu().numpy().flatten()

# --- Repeat Step 3 & 4 for testing dataset ---
df_testing = pd.read_csv("testing_data.csv")
X_all_testing = df_testing[['ncpus', 'mem_gb', 'submit_hour', 'queue_type']].values
y_actual_testing = df_testing['actual_runtime'].values

X_all_testing_tensor = torch.tensor(X_all_testing, dtype=torch.float32).to(device)

with torch.no_grad():
    y_predicted_testing = model(X_all_testing_tensor).cpu().numpy().flatten()

# --- Step 5: Build the full comparison table ---
comparison_training = pd.DataFrame({
    "ncpus": df_training['ncpus'].values,
    "mem_gb": df_training['mem_gb'].values,
    "submit_hour": df_training['submit_hour'].values,
    "queue_type": df_training['queue_type'].values,
    "actual_runtime": y_actual_training,
    "predicted_runtime": y_predicted_training,
    "abs_error": abs(y_actual_training - y_predicted_training)
})

comparison_testing = pd.DataFrame({
    "ncpus": df_testing['ncpus'].values,
    "mem_gb": df_testing['mem_gb'].values,
    "submit_hour": df_testing['submit_hour'].values,
    "queue_type": df_testing['queue_type'].values,
    "actual_runtime": y_actual_testing,
    "predicted_runtime": y_predicted_testing,
    "abs_error": abs(y_actual_testing - y_predicted_testing)
})

# --- Step 6: Display only a readable preview ---
print("=== First 10 rows of Training (preview) ===")
print(comparison_training.head(10).to_string(index=False))

print("\n=== First 10 rows of Testing (preview) ===")
print(comparison_testing.head(10).to_string(index=False))

# --- Step 7: Save the FULL comparison to a file, for later inspection if needed ---
comparison_training.to_csv("full_predictions_training.csv", index=False)
print(f"\nFull comparison for all {len(df_training)} rows saved to full_predictions_training.csv")

comparison_testing.to_csv("full_predictions_testing.csv", index=False)
print(f"Full comparison for all {len(df_testing)} rows saved to full_predictions_testing.csv")

# --- Step 8: Accuracy metrics calculated across the ENTIRE dataset ---
mae = comparison_training['abs_error'].mean()
mean_actual = y_actual_training.mean()
accuracy_pct = 100 * (1 - (mae / mean_actual))

print("\nTesting Dataset:")
print(f"Mean Absolute Error (all {len(df_training)} rows): {mae:.2f}")
print(f"Approximate accuracy: {accuracy_pct:.2f}%")

mae = comparison_testing['abs_error'].mean()
mean_actual = y_actual_testing.mean()
accuracy_pct = 100 * (1 - (mae / mean_actual))

print("\nTraining Dataset:")
print(f"Mean Absolute Error (all {len(df_testing)} rows): {mae:.2f}")
print(f"Approximate accuracy: {accuracy_pct:.2f}%")