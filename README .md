
# GNNModel

This repository contains the implementation of a Graph Neural Network (GNN) using PyTorch and PyTorch Geometric for analyzing and predicting data. The program processes data from a provided database and trains a GNN model on it.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [License](#license)

## Installation

To run this program, ensure you have the following libraries installed. If not, you can install them using pip:

```bash
pip install pandas torch torch-geometric sklearn matplotlib
```

## Usage

1. **Extract Data**: Ensure the data zip file is extracted to the appropriate directory specified in the code.

2. **Run the Script**: Execute the Python script to load data, preprocess it, and train the GNN model.

```bash
python GNNModel7.py
```

## File Structure

- `GNNModel7.py`: Main script containing the implementation of data loading, preprocessing, model training, and evaluation.
- `data/`: Directory where the data files are stored.

## Dependencies

- pandas
- numpy
- sklearn
- torch
- torch-geometric
- matplotlib
- sqlite3
- zipfile
- os

## Data Preparation

The data is expected to be in a zip file, which is extracted to a specified folder. The main database file is then loaded from the extracted folder. The script includes code to handle the extraction and loading process.

```python
# Example code to extract and load data
zip_file_path = 'path/to/data.zip'
extracted_folder_path = 'path/to/extracted_data'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

overview_db_path = os.path.join(extracted_folder_path, 'data.db')
conn = sqlite3.connect(overview_db_path)
```

## Training the Model

The model is defined using PyTorch and PyTorch Geometric. The script includes steps for data preprocessing, defining the model architecture, training the model, and evaluating its performance.

```python
# Example model definition
class GNNModel(nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## Evaluation

The model's performance is evaluated using appropriate metrics and visualizations, such as accuracy and loss plots.

```python
# Example evaluation code
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
