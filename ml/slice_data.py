"""
This module outputs the performance of the model on slices of the data for a selected categorical feature.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compute_model_metrics, inference
import joblib
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

with open("../config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

cat_features = config['cat_features']
slice_feature = config['slice_feature']

def get_sliced_data_performance():
    logging.info("Loading data")
    data = pd.read_csv("../data/census_cleaned.csv")
    _, test = train_test_split(data, test_size=0.20)

    model = joblib.load('../model/model.pkl')
    encoder = joblib.load('../model/encoder.pkl')
    lb = joblib.load('../model/lb.pkl')

    slice_metrics = []
    for cat in test[slice_feature].unique():
        temp = test[test[slice_feature] == cat]

        X_test, y_test, _, _ = process_data(
            temp,
            cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False)

        y_pred = model.predict(X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        row = f"{slice_feature} - {cat} Precision: {precision: .2f}, recall: {recall: .2f}, fbeta: {fbeta: .2f}"
        slice_metrics.append(row)

        with open('../sliced_data_metrics/slice_output.txt', 'w') as file:
            for row in slice_metrics:
                file.write(row + '\n')

    logging.info("Performance metrics for sliced data are saved to slice_output.txt")


if __name__ == '__main__':
    get_sliced_data_performance()