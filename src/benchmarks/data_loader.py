import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


def load_data(dataset):
    if dataset == "adult":
        return __load_adult("data/adult.data")
    raise ValueError("Dataset " + dataset + " doesn't exist.")


def __load_adult(filepath):
    headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "income"]
    categorical_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                           "native-country"]

    data = pd.read_csv(filepath, sep=r"\s*,\s*", engine="python", names=headers, na_values="?")

    # use one hot encoding
    data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns, dummy_na=True)
    # drop rows with missing values, 30162 rows left
    data = data.dropna(how="any")
    for category in categorical_columns:
        data = data.drop(data[data[category + "_nan"] == 1].index)
    # delete NaN-dummy-columns
    for category in categorical_columns:
        del data[category + "_nan"]

    # make 'income' column the last one
    columns = data.columns.tolist()
    income_column_index = data.columns.get_loc("income")
    columns = columns[:income_column_index] + columns[income_column_index + 1:] + [columns[income_column_index]]
    data = data[columns]

    # replace 'income' strings with integers for classification
    data["income"] = data["income"].replace({">50K": -1, "<=50K": 1})

    data = data.to_numpy()

    X = np.delete(data, -1, 1)
    y = data[:, -1]

    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    return X, y
