# Reliable Classification of Imbalanced Lung Cancer Data with Enhanced Split Conformal Prediction (ESCP)

The goal of the paper is to enhance the classification performance and predictive reliability of lung cancer data, particularly when dealing with high-dimensional, imbalanced gene expression data. The paper proposes an ***Enhanced Split Conformal Prediction (ESCP)*** framework that addresses the challenges of traditional classification methods. Key Features of the Approach:

Feature Selection: The paper uses ***Sure Independence Screening (SIS)*** to select relevant features strongly associated with lung cancer status, helping to mitigate the effects of high-dimensionality in the data.

Class Imbalance Handling: The ***SMOTE (Synthetic Minority Over-sampling Technique)*** is applied to balance the dataset by generating synthetic minority class samples, which improves the recognition of minority subtypes of lung cancer.

Conformal Prediction: The framework integrates the ***Split Conformal Prediction (SCP)*** method to provide statistical guarantees on the prediction coverage, ensuring the reliability of predictions with confidence intervals.

The ***ESCP*** method significantly improves both the accuracy and reliability of classification models by addressing class imbalance and quantifying the uncertainty of predictions, making it particularly useful for clinical applications where accurate decision-making is critical.

***Overview***
This repository implements the Enhanced Split Conformal Prediction (ESCP) framework for reliable classification of imbalanced lung cancer datasets. The ESCP algorithm combines several advanced techniques, including Sure Independence Screening (SIS), Synthetic Minority Over-sampling Technique (SMOTE), and Split Conformal Prediction (SCP), to address the challenges of high-dimensional and imbalanced medical data. It aims to improve classification accuracy, especially for minority classes, and provide statistically reliable predictions for clinical applications.

## Key Features
Feature Selection: Utilizes Sure Independence Screening (SIS) to reduce data dimensionality and select relevant features for classification.

Class Imbalance Handling: Implements SMOTE to oversample minority class samples and balance the dataset, improving recognition of rare cancer subtypes.

Reliable Predictions: Integrates Split Conformal Prediction (SCP) to generate prediction sets with statistical guarantees, offering a confidence interval for each prediction.

Performance Evaluation: Demonstrates superior performance on multiple lung cancer gene expression datasets, achieving high AUROC, AUPRC, and coverage.

## Methodology
***Data Preprocessing:***

Variance Filtering: Removes features with low variance to reduce noise.

Sure Independence Screening (SIS): Selects the most relevant features for classification.

Class Imbalance Handling:

SMOTE: Generates synthetic minority samples to balance the dataset.

## Model Training:

A base classifier (e.g., XGBoost, SVM, Random Forest) is trained using the balanced dataset.

Split Conformal Prediction:

SCP calculates nonconformity scores and generates prediction sets for each sample, providing confidence intervals for predictions.

## Evaluation:

ESCP is evaluated on eight public lung cancer gene expression datasets (IM101-IM108), showing improved classification and prediction reliability compared to traditional methods.

## Datasets
The framework is evaluated on the following datasets:

Lung Genedata (IM101)

Complete Dataframe (IM102)

miRNA Lung (IM103)

KIPC LUAD (IM106)

ICGC LUAD (IM107)

PRAD LUAD (IM108)

Each dataset contains high-dimensional gene expression data with imbalanced classes, reflecting the challenges in lung cancer classification tasks.

Performance Metrics
The performance of the model is evaluated using the following metrics:

Accuracy: Overall classification accuracy.

F1-Score: Harmonic mean of precision and recall.

AUROC: Area Under the Receiver Operating Characteristic Curve.

AUPRC: Area Under the Precision-Recall Curve.

Coverage: The proportion of true values captured within the prediction set.

## Results
The ESCP framework outperforms traditional methods (e.g., SVM, Random Forest) in handling imbalanced and high-dimensional data. Key results include:

High AUROC and AUPRC: Demonstrated better performance in recognizing minority classes.

Reliable Confidence Intervals: Empirical coverage rates closely match the pre-specified confidence level (e.g., 96.6% coverage on IM106).

Robustness to Overfitting: ESCP handles ultra-high dimensional datasets effectively, maintaining high coverage and stability.
