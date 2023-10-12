# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model is Random Forest classifier from the scikit-learn library with hyperparameters set in config.yaml.

## Intended Use
Predicting if the person's income exceeds $50K/yr based on census data.

## Training Data
The training data set is Census data set from UCI data base.

## Evaluation Data
80-20 split was used to split train and test data with fixed random seed.

## Metrics
The model was evaluated with the following metrics and here are the performance values: Precision: 0.78. Recall: 0.56. Fbeta: 0.65.

## Ethical Considerations
Data has gender and race features, therefore the performance on these categories should be investigated.

## Caveats and Recommendations
Trying other models and model ensembling could result in better performance.
