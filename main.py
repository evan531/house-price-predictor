import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_encoded_data(path_to_file):
    """
    Encodes categorical features of given csv file with one hot encoding
    Concatonates the one hot features the the original data
    Drops the categorical features and Id from the data

    params:
        path_to_file: string - path to csv file

    return: DataFrame

    """

    data = pd.read_csv(path_to_file)

    categorical_columns = data.select_dtypes(include="object").columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)

    one_hot_encoded = encoder.fit_transform(data[categorical_columns])

    one_hot_data = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    data_encoded = pd.concat([data, one_hot_data], axis=1)

    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    data_encoded = data_encoded.drop("Id", axis=1)
    return data_encoded


train_data = get_encoded_data("./data/train.csv")


# device = "cuda" if torch.cuda.is_available() else "cpu"


# class CSVDataset(Dataset):
#     def __init__(self, csv_file):
#         self.data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
#         self.numerical_data =
#         self.data = torch.tensor(self.data, dtype=torch.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# dataset = CSVDataset("./data/test.csv")

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# for batch in dataloader:
#     print(batch)
