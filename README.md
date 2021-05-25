# Normalization FeatureCloud App

## Description
A Normalization FeautureCloud app, allowing to perform different normalization and standardization techniques in a federated manner.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data

## Output
- train_normalized.csv containing the normalized training data 
- test_normalized.csv containing the normalized test data
- train.csv containing the original, unnormalized, local training data
- test.csv containing the original, unnormalized, local test data

## Workflows
Can be combined with the following apps:
- Pre: Cross Validation, Feature Selection
- Post: Various Analysis apps (e.g. Random Forest, Linear Regression, Logistic Regression, ...)

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_normalization:
  input:
    train: "train.csv"
    test: "test.csv"
  output:
    pred: "pred.csv"
    proba: "proba.csv"
    test: "test.csv"
  format:
    sep: ","
    label: "Class" # the label column is excluded from the normalization
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
  normalization: variance
