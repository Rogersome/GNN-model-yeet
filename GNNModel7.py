# Step 1: Install Required Libraries
# If not already installed, uncomment the following line to install the necessary libraries.
# !pip install pandas torch torch-geometric sklearn matplotlib

# Step 2: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sqlite3
import zipfile
import os
import matplotlib.pyplot as plt

# Step 3: Load and Prepare Data
# Load and preprocess the dataset from the provided path
zip_file_path = 'Cleaned-20240412T013610Z-001.zip'

# Extract the zip file
extracted_folder_path = 'D:/Jeff/GP/Cleaned_Data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# Load the Overview database
overview_db_path = os.path.join(extracted_folder_path, 'Cleaned', 'Overview.db')
conn = sqlite3.connect(overview_db_path)
overview_df = pd.read_sql_query("SELECT * FROM Overview;", conn)
conn.close()

# Calculate SoH as the ratio of the current capacity to the initial capacity
initial_capacity = overview_df['Capacity'].iloc[0]
overview_df['SoH'] = overview_df['Capacity'] / initial_capacity

# Features and target
features = overview_df[['Cycles_or_Days_of_Aging_Test', 'ten_seconds_Pulse_Resistance', 'Capacity']]
target = overview_df['SoH']

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert to PyTorch tensors
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(target.values, dtype=torch.float).view(-1, 1)

# Create edge index (for simplicity, we will create a fully connected graph)
num_nodes = x.size(0)
edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

# Create PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y)
data_list = [data]

# Create DataLoader
loader = DataLoader(data_list, batch_size=1)

# Step 4: Define the Improved GNN Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        return x

# Define model, loss, and optimizer
input_dim = features.shape[1]
hidden_dim = 64
output_dim = 1

model = GNNModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.MSELoss()

# Step 5: Train the Model
# Training loop with loss tracking
epochs = 500
losses = []

model.train()
for epoch in range(epochs):  # Number of epochs
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(loader))
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss/len(loader):.4f}')

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Make Predictions with User Input and Visualize Impact of Cycles_or_Days_of_Aging_Test
def predict_soh(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(scaler.transform([input_data]), dtype=torch.float)
        
        # Create a simple edge index for a single node
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        input_data = Data(x=input_tensor, edge_index=edge_index)
        output = model(input_data)
        return output.item()

# Function to evaluate the impact of Cycles_or_Days_of_Aging_Test
def evaluate_impact_on_soh(model, cycles_range, fixed_input):
    soh_predictions = []
    for cycles in cycles_range:
        input_data = [cycles] + fixed_input[1:]
        predicted_soh = predict_soh(model, input_data)
        soh_predictions.append(predicted_soh)
    return soh_predictions

# Function to evaluate the SoH
def evaluate_soh(soh):
    threshold = 0.8  # You can adjust this threshold as needed
    if soh >= threshold:
        return "Good"
    else:
        return "Bad"

# Define a range of Cycles_or_Days_of_Aging_Test values
cycles_range = np.arange(0, 300, 10)

# Define fixed values for the other inputs (ten_seconds_Pulse_Resistance and Capacity)
fixed_input = [0, 0.0013, 55.0]

# Get the SoH predictions for the range of cycles
soh_predictions = evaluate_impact_on_soh(model, cycles_range, fixed_input)

# Plot the impact of Cycles_or_Days_of_Aging_Test on SoH
# plt.figure(figsize=(10, 6))
# plt.plot(cycles_range, soh_predictions, label='SoH Prediction')
# plt.xlabel('Cycles_or_Days_of_Aging_Test')
# plt.ylabel('Predicted SoH')
# plt.title('Impact of Cycles_or_Days_of_Aging_Test on SoH')
# plt.legend()
# plt.grid(True)
# plt.show()

# Step 7: User input for real-time prediction
Cycles_or_Days_of_Aging_Test = float(input("Enter your battery's cycles or Days of Aging Test: "))
ten_seconds_Pulse_Resistance = float(input("Enter your battery's ten seconds Pulse Resistance: "))
Capacity = float(input("Enter your battery's capacity: "))
user_input = [Cycles_or_Days_of_Aging_Test, ten_seconds_Pulse_Resistance, Capacity]
predicted_soh = predict_soh(model, user_input)
health_status = evaluate_soh(predicted_soh)
print(f'Predicted SoH: {predicted_soh:.4f} - Battery Health: {health_status}')
