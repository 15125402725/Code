# Reliable Classification of Imbalanced Lung Cancer Data with Enhanced Split Conformal Prediction (ESCP)
This repository implements the ***Enhanced Split Conformal Prediction (ESCP)*** framework for reliable classification of imbalanced lung cancer datasets. The ESCP algorithm combines several advanced techniques, including ***Sure Independence Screening (SIS)***, ***Synthetic Minority Over-sampling Technique (SMOTE)***, and ***Split Conformal Prediction (SCP)***, to address the challenges of high-dimensional and imbalanced medical data. It aims to improve classification accuracy, especially for minority classes, and provide statistically reliable predictions for clinical applications.

## Code Overview
This repository contains a pipeline for preprocessing, feature selection, and predictive modeling on high-dimensional gene expression datasets. The workflow is modularized into several scripts:

***1.Target_Variable_8.py – Variance Filtering***

Removes low-variance features from the raw dataset using a threshold-based filter.

Input: original dataset (.csv) with target variable in the first column.

Output: filtered dataset (*_filtered_gene_data.csv).

***2.SIS_8.py – Feature Selection (Sure Independence Screening / ANOVA F-test)***

Selects the top-K features most associated with the target variable.

Input: variance-filtered dataset.

Output: SIS-selected dataset (*_filtered_gene_data_SIS.csv).

***3.Modeling and Conformal Prediction***

Several scripts implement predictive modeling and uncertainty quantification using different classifiers:

RF.py-Random Forest

RFS.py-Random Forest with SMOTE balancing and 5-fold cross-validation

RFSCP.py – Random Forest + Split Conformal Prediction (SCP)

SVM.py-Support Vector Machine (SVM

SVMS.py-SVM with SMOTE balancing and 5-fold cross-validation

SVMSCP.py – Support Vector Machine (SVM) + SCP

XGBoost.py-XGBoost

XGBS.py – XGBoost with SMOTE balancing and 5-fold cross-validation

XGBSCP.py – XGBoost + SCP


These scripts train models on the SIS-selected dataset, evaluate performance (Accuracy, F1-score, AUC, etc.), and visualize results (ROC/PR curves, confusion matrices, calibration plots).

Outputs are saved in results/ or model_evaluation_plots/ directories.
## Data Information:
This study used eight public lung cancer gene expression datasets from Kaggle and UCI, each with different features and scales, suitable for evaluating classification performance on high-dimensional imbalanced datasets. Data preprocessing and feature selection were performed using variance filtering and Sure Independence Screening (SIS) methods, removing low-variance features and retaining those highly associated with the target variable.

The specific characteristics of the datasets are as follows:

***Lung genedata（IM101）:*** Focuses on lung squamous cell carcinoma (LUSC). Contains data from 551 patients, 321
each with 56,907 TPM-normalized gene expressions. Class imbalance: 502 cancer vs. 49 healthy. Used 322
for analyzing LUSC gene features and classification. 323

***complete dataframe（IM102）:*** From five medical centers, includes 442 samples with over 23,000 gene 324
expressions.The target is survival for more than 18 months; high-risk is defined based on this criterion.. 325
High feature dimension, suitable for feature selection to avoid overfitting. 326

***miRNA lung（IM103）:*** miRNA data for small cell (SCLC) and non-small cell lung cancer (NSCLC). 119 327
NSCLC and 49 SCLC cell lines, with 743 miRNA features. 328

***data（IM104）, KIPC LUAD(IM106), PRAD LUAD(IM108):***  From ICMR, includes multiple cancer types (breast, kidney, 329
colon, lung). 802 samples, each with over 20,000 gene expressions. Used for multi-class classification of 330
cancer types. 331

***Count matrix(IM105):*** 60,660 genes across 600 samples—317 normal, 283 lung cancer. 332
***icgc LUAD(IM107):*** Lung adenocarcinoma data; 543 lung cancer and 55 normal samples, 19,565 genes

## [data] (https://github.com/15125402725/Data)

Each dataset contains high-dimensional gene expression data with imbalanced classes, reflecting the challenges in lung cancer classification tasks.

## Code information
The processing of these eight datasets in this study is as follows:

First, variance filtering was applied for preprocessing; then the SIS method was used to select features that are highly associated with our target variables (e.g., high-risk vs. low-risk, SCLC vs. NSCLC, lung cancer vs. other common cancers, etc.).[Code]  (https://github.com/15125402725/code);Finally,Modeling and Evaluation.

## Usage Instuctions
Variance Filtering → *_filtered_gene_data.csv

SIS Feature Selection → *_filtered_gene_data_SIS.csv

Model Training + SCP → evaluation metrics and visualizations
## Requirements-Any dependencies
pandas>=1.3

numpy>=1.20

scikit-learn>=0.24

xgboost>=1.5

imbalanced-learn>=0.8

matplotlib>=3.4

seaborn>=0.11

## Methodology
***Data Processing***

Eight publicly available lung cancer gene expression datasets from Kaggle and UCI were used.

Variance filtering was applied to remove low-variance features, and SIS/ANOVA F-test was employed to select the features most relevant to the target variable.

SMOTE was applied to the training set to balance class distribution.

***Modeling and Evaluation***

Three classifiers were trained on the selected features: Random Forest, Support Vector Machine (SVM), and XGBoost.

Stratified train-test split and five-fold cross-validation were used for evaluation, with metrics including Accuracy, F1, ROC-AUC, Average Precision, and G-mean.

To quantify predictive uncertainty, Split Conformal Prediction (SCP) was introduced, analyzing the trade-off between coverage and prediction set size under different confidence levels.
## quotation
In this study, we utilized the icgc LUAD dataset for lung cancer prediction. The gene expression data from this dataset was used to train and evaluate deep learning models. Specifically, Liu S and Yao W (2022) proposed a deep learning method with KL divergence gene selection based on gene expression to improve the predictive accuracy of lung cancer.

# Reference:

Liu S, Yao W. Prediction of lung cancer using gene expression and deep learning with KL divergence gene selection. BMC Bioinformatics, 2022, 23(1): 175.

## Materials and Methods
### Computational Infrastructure:
python python 3.12 Platform: x86_64-w64-mingw32/x64 (64-bit) Running under: Windows >= 10 x64 (build 26100) The packages loaded:numpy_1.19.0, pandas_1.1.0, scikit-learn_0.24.0,xgboost_1.3.0, imbalanced-learn_0.8.0, matplotlib_3.3.0, seaborn_0.11.0




