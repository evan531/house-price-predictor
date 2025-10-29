import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")


def get_encoded_data(dataset: pd.DataFrame, train: bool):
    """
    Encodes categorical features of given csv file with one hot encoding
    Concatonates the one hot features the the original data
    Drops the categorical features and Id from the data

    params:
        dataset: pd.DataFrame - pandas DataFrame containing the entire dataset
        train: bool - fits the encoder if train is true, otherwise uses the encoder already fitted

    return:
        num_data: pandas.Dataframe - Contains the numerical data
        cat_data: pandas.Datafram - Contains the one-hot encoded categorical data

    """

    categorical_columns = dataset.select_dtypes(include="object").columns.tolist()

    cat_data = []

    if train:
        cat_data = encoder.fit_transform(dataset[categorical_columns])
    else:
        cat_data = encoder.transform(dataset[categorical_columns])

    one_hot_data = pd.DataFrame(
        cat_data, columns=encoder.get_feature_names_out(categorical_columns)
    )

    data_encoded = pd.concat([dataset, one_hot_data], axis=1)

    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    data_encoded = data_encoded.drop("Id", axis=1)

    num_data = dataset.drop(categorical_columns, axis=1)
    num_data = num_data.drop("Id", axis=1)

    num_data.fillna(value=0, inplace=True)

    return num_data, one_hot_data


data = pd.read_csv("./data/train.csv")


# Get the numerical data and one-hot-coded categorical data in the form of pandas dataframes
num_data, cat_data = get_encoded_data(data, train=True)

# Extract the sale price as the target
train_targets = num_data.get("SalePrice")

# Drop sale price from feature set
num_data = num_data.drop("SalePrice", axis=1)

# Normalized the numerical data between 1 and 0
num_data = pd.DataFrame(MinMaxScaler().fit_transform(num_data))

# Concatonate one-hot-encoded data with numerical data
train_data = pd.concat([num_data, cat_data], axis=1)

# Split into train and validation data
train_data, valid_data, train_targets, valid_targets = train_test_split(
    train_data, train_targets, train_size=0.8, shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Convert matrices and vectors into pytorch tensors
train_data = torch.tensor(train_data.values, dtype=torch.float32).to(device)
train_targets = (
    torch.tensor(train_targets.values, dtype=torch.float32).reshape(-1, 1).to(device)
)
valid_data = torch.tensor(valid_data.values, dtype=torch.float32).to(device)
valid_targets = (
    torch.tensor(valid_targets.values, dtype=torch.float32).reshape(-1, 1).to(device)
)

"""
Regression model comprising of 2 hidden layer.
Uses batch normalization after input layer and first hidden layer.
Utilize dropout to prevent over parameterization.
"""
model = nn.Sequential(
    nn.Linear(303, 256),
    nn.BatchNorm1d(256, eps=1e-4),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128, eps=1e-4),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

# Huber loss converged the fastest and most accurate when compared to L1 and MSE loss.
loss_fn = nn.HuberLoss()

# Adam and AdaGrad performed similar, but Adam converged faster
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)

# Scheduler is key to acheiving lowest error rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


# Set hyper params
epochs = 100
batch_size = 128
batch_start = torch.arange(0, len(train_data), batch_size)

min_error = np.inf
best_weights = None
validation_history = []
training_history = []

"""
Train and test the model on validation error every epoch.
Save the lowest error and the weights associated with it.
"""
for epoch in range(epochs):
    model.train()

    for batch in batch_start:
        data_batch = train_data[batch : batch + batch_size].to(device)

        targets_batch = train_targets[batch : batch + batch_size].to(device)
        targets_prediction = model(data_batch)
        loss = loss_fn(targets_prediction, targets_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    model.eval()
    with torch.no_grad():
        training_prediction = model(train_data)
        valid_prediction = model(valid_data)

        validation_error = np.abs(
            np.mean(valid_prediction.cpu().numpy() - valid_targets.cpu().numpy())
        )
        training_error = np.abs(
            np.mean(training_prediction.cpu().numpy() - train_targets.cpu().numpy())
        )
        print(f"Current error: {validation_error}")
        if validation_error < min_error:
            min_error = validation_error
            best_weights = copy.deepcopy(model.state_dict())

        validation_history.append(validation_error)
        training_history.append(training_error)

# Save model with lowest loss
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "./models/huber_loss_with_adam.pt")

# Load test data and encode it
test_data = pd.read_csv("./data/test.csv")
test_num_data, test_cat_data = get_encoded_data(test_data, train=False)

test_num_data = pd.DataFrame(MinMaxScaler().fit_transform(test_num_data))

test_df = pd.concat([test_num_data, test_cat_data], axis=1)

predictions_df = pd.DataFrame()
predictions_df["Id"] = test_data.get("Id").values

# Convert to tensor
test_df = torch.tensor(test_df.to_numpy(), dtype=torch.float32).to(device)

model.eval()

# Make predictinos and save to file
with torch.no_grad():
    predictions = model(test_df)
    predictions_df["SalePrice"] = predictions.cpu().numpy()
    predictions_df.to_csv("./test_target.csv", index=False)

# Plot validation and training loss
plt.plot(validation_history, label="Validation Error")
plt.plot(training_history, label="Training Error")
plt.legend()
plt.show()
