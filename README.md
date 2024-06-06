# Breast Cancer Wisconsin (Diagnostic) Data Set ML Models

This repository contains two machine learning models trained on the Breast Cancer Wisconsin (Diagnostic) Data Set.

## Model 1: Logistic Regression
- **File:** Mubarak_model_v1.py
- **Description:** Logistic Regression model for classifying breast cancer as malignant or benign. This model preprocesses the data by standardizing the features and trains a logistic regression classifier.

## Model 2: Decision Tree
- **File:** Mubarak_model_v2.py
- **Description:** Decision Tree model for classifying breast cancer as malignant or benign. This model preprocesses the data similarly to the logistic regression model but uses a decision tree classifier.

## Dataset
The dataset used for training both models is the Breast Cancer Wisconsin (Diagnostic) Data Set, which is publicly available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

## Instructions
1. **Model 1:** To run the logistic regression model, execute `Mubarak_model_v1.py`.
2. **Model 2:** To run the decision tree model, execute `Mubarak_model_v2.py`.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install the required packages using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
