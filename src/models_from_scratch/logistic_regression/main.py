import numpy as np
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
from utils import train_test_split

from models_from_scratch.logistic_regression.logistic_regressor import LogisticRegressor

def load_dataset():
    from sklearn.datasets import fetch_openml

    data = fetch_openml("adult", version=2)
    return data["frame"]

def preprocessing(data: pd.DataFrame):
    y = data.pop("class")
    X = data
    category_columns = X.select_dtypes("category")
    dummy_array = []
    for column in category_columns:
        dummy_columns = pd.get_dummies(X[column], dtype=np.int8, drop_first=True)
        X.pop(column)
        dummy_array.append(dummy_columns)
    dummy_array.insert(0, X)
    X = pd.concat(dummy_array, axis=1)

    labels = y.cat.categories
    y = y.apply(lambda x: labels.get_loc(x)).astype(int)

    assert X.isna().sum().sum() == 0 and y.isna().sum() == 0, "nan"

    return X.values, y.values, labels

def main():
    data = load_dataset()
    print(data.isna().sum())

    no_work = data.query("workclass.isna() and occupation.isna()")
    print(no_work["class"].value_counts())
    print(data["class"].value_counts())
    print(data.query("`native-country`.isna()")["class"].value_counts())

    data = data.query("~workclass.isna() and ~occupation.isna() and ~`native-country`.isna()")
    data = data.drop(columns=["education", "fnlwgt", "sex", "race"])
    X, y, labels = preprocessing(data)

    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, 0.8, shuffle=True)
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_std = np.where(X_train_std == 0, 1, X_train_std)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    regressor = LogisticRegressor(learning_rate=0.1)
    regressor.fit(X_train, y_train, epochs=500)
    print(regressor.evaluate(X_test, y_test))

if __name__ == "__main__":
    main()