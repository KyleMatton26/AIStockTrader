import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
import ast

print(torch.__version__)
print("Cuda available:", torch.cuda.is_available())

# 1. Data (preparing and loading)

# Load the dataset from a file
def load_dataset_from_file(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Safely evaluate the line to a tuple
                item = ast.literal_eval(line)
                dataset.append(item)
    return dataset

dataset = load_dataset_from_file("combined_X_dataset.txt")

random.shuffle(dataset)

X_data = []
Y_labels = []

def get_data_and_labels(dataset):
    for item in dataset:
        X_data.append(item[0])
        Y_labels.append(item[1])


get_data_and_labels(dataset)

label_map = {
    'positive': 1,
    'negative': -1,
    'neutral': 0
}

Y_labels_numeric = []
for label in Y_labels:
    Y_labels_numeric.append(label_map[label])

for i in range(5):
    print(X_data[i])
    print(Y_labels[i])
    print(Y_labels_numeric[i])
    print("------------")

def split_data(X_data, Y_labels, test_size, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_labels, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, Y_train, Y_test
    
test_size = 0.2  # 20% of the data will be used for testing

# Split the dataset
X_train, X_test, Y_train, Y_test = split_data(X_data, Y_labels_numeric, test_size)

# Print out the size of each split to verify
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

print(X_train[0])
print(Y_train[0])


# 2. Build Model (Possibly an LSTM model)


# 3. Fitting the model to data (training) --- MAKE SURE TO SET MANUAL_SEED


# 4. Making predictions and evaluating a model (inference)


# 5. Saving and loading a model


# 6. Putting it all together