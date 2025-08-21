# Reliable Classification of Imbalanced Lung Cancer Data with Enhanced Split Conformal Prediction (ESCP)

The goal of the paper is to enhance the classification performance and predictive reliability of lung cancer data, particularly when dealing with high-dimensional, imbalanced gene expression data. The paper proposes an ***Enhanced Split Conformal Prediction (ESCP)*** framework that addresses the challenges of traditional classification methods. Key Features of the Approach:

Feature Selection: The paper uses ***Sure Independence Screening (SIS)*** to select relevant features strongly associated with lung cancer status, helping to mitigate the effects of high-dimensionality in the data.

Class Imbalance Handling: The ***SMOTE (Synthetic Minority Over-sampling Technique)*** is applied to balance the dataset by generating synthetic minority class samples, which improves the recognition of minority of lung cancer.

Conformal Prediction: The framework integrates the ***Split Conformal Prediction (SCP)*** method to provide statistical guarantees on the prediction coverage, ensuring the reliability of predictions with confidence intervals.

The ***ESCP*** method significantly improves both the accuracy and reliability of classification models by addressing class imbalance and quantifying the uncertainty of predictions, making it particularly useful for clinical applications where accurate decision-making is critical.

***Overview:***
This repository implements the ***Enhanced Split Conformal Prediction (ESCP)*** framework for reliable classification of imbalanced lung cancer datasets. The ESCP algorithm combines several advanced techniques, including ***Sure Independence Screening (SIS)***, ***Synthetic Minority Over-sampling Technique (SMOTE)***, and ***Split Conformal Prediction (SCP)***, to address the challenges of high-dimensional and imbalanced medical data. It aims to improve classification accuracy, especially for minority classes, and provide statistically reliable predictions for clinical applications.

## Datasets
## Data Description:
This study used eight public lung cancer gene expression datasets, each with different features and scales, suitable for evaluating classification performance on high-dimensional imbalanced datasets. Data preprocessing and feature selection were performed using variance filtering and Sure Independence Screening (SIS) methods, removing low-variance features and retaining those highly associated with the target variable.
Libraries Used: This study used Python libraries such as pandas, sklearn ,sklearn.feature_selection, scikit-learn, XGBoost, SMOTE, and matplotlib.

***Lung genedata:*** Focuses on lung squamous cell carcinoma (LUSC). Contains data from 551 patients, 321
each with 56,907 TPM-normalized gene expressions. Class imbalance: 502 cancer vs. 49 healthy. Used 322
for analyzing LUSC gene features and classification. 323

***complete dataframe:*** From five medical centers, includes 442 samples with over 23,000 gene 324
expressions.The target is survival for more than 18 months; high-risk is defined based on this criterion.. 325
High feature dimension, suitable for feature selection to avoid overfitting. 326

***miRNA lung:*** miRNA data for small cell (SCLC) and non-small cell lung cancer (NSCLC). 119 327
NSCLC and 49 SCLC cell lines, with 743 miRNA features. 328

***data, KIPC LUAD, PRAD LUAD:***  From ICMR, includes multiple cancer types (breast, kidney, 329
colon, lung). 802 samples, each with over 20,000 gene expressions. Used for multi-class classification of 330
cancer types. 331

***Count matrix:*** 60,660 genes across 600 samples—317 normal, 283 lung cancer. 332
***icgc LUAD:*** Lung adenocarcinoma data; 543 lung cancer and 55 normal samples, 19,565 genes
## [data] (https://github.com/15125402725/Data)
The framework is evaluated on the following datasets:

Lung Genedata (IM101)

Complete Dataframe (IM102)

miRNA Lung (IM103)

data(IM104)

Count matrix(IM105)

KIPC LUAD (IM106)

icgc LUAD (IM107)

PRAD LUAD (IM108)

Each dataset contains high-dimensional gene expression data with imbalanced classes, reflecting the challenges in lung cancer classification tasks.


## Materials and Methods
### Computational Infrastructure:
***software:*** python python 3.13 Platform: x86_64-w64-mingw32/x64 (64-bit) Running under: Windows >= 10 x64 (build 26100) The packages loaded:numpy_1.19.0, pandas_1.1.0, scikit-learn_0.24.0,xgboost_1.3.0, imbalanced-learn_0.8.0, matplotlib_3.3.0, seaborn_0.11.0
## [Code]  (https://github.com/15125402725/code)
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


## Performance Metrics
The performance of the model is evaluated using the following metrics:

***Accuracy:*** Overall classification accuracy.

***F1-Score:*** Harmonic mean of precision and recall.

***AUROC:*** Area Under the Receiver Operating Characteristic Curve.

***AUPRC:*** Area Under the Precision-Recall Curve.

***Coverage:*** The proportion of true values captured within the prediction set.

## Results
The ESCP framework outperforms traditional methods (e.g., SVM, Random Forest) in handling imbalanced and high-dimensional data. Key results include:

High AUROC and AUPRC: Demonstrated better performance in recognizing minority classes.

Reliable Confidence Intervals: Empirical coverage rates closely match the pre-specified confidence level (e.g., 96.6% coverage on IM106).

Robustness to Overfitting: ESCP handles ultra-high dimensional datasets effectively, maintaining high coverage and stability.

## quotation
In this study, we utilized the icgc LUAD dataset for lung cancer prediction. The gene expression data from this dataset was used to train and evaluate deep learning models. Specifically, Liu S and Yao W (2022) proposed a deep learning method with KL divergence gene selection based on gene expression to improve the predictive accuracy of lung cancer.

***Reference:***

Liu S, Yao W. Prediction of lung cancer using gene expression and deep learning with KL divergence gene selection. BMC Bioinformatics, 2022, 23(1): 175.
