# 1.Title - HOCP: Hybrid optimization and conformal prediction for early Alzheimer's detection under class-imbalance and uncertainty

***HOCP (Hybrid Optimization and Conformal Prediction)*** is a systematic framework for the early prediction of Alzheimer’s disease (AD), designed to simultaneously address two major challenges commonly encountered in clinical data: class imbalance, where patients with AD constitute the minority class, and the inability of conventional models to provide reliable uncertainty estimates because they typically produce only deterministic class labels. The framework addresses these challenges through three complementary levels. At the data level, LASSO regression is employed for feature selection, while SMOTE oversampling is performed within each fold of 10 fold cross validation to balance the class distribution of the training data. At the algorithmic level, a hybrid hyperparameter optimization strategy, termed Grid CPSO, is proposed. Grid search is first used to perform a global coarse search and identify promising regions of the hyperparameter space, after which chaotic particle swarm optimization is applied to conduct a more refined local search. This strategy is intended to improve minority class recognition and enhance model generalization stability. At the uncertainty quantification level, Mondrian conformal prediction is incorporated to generate statistically guaranteed prediction sets for each test sample rather than a single class label, thereby providing class conditional coverage guarantees for patients with AD at a 95% confidence level. In addition, the framework integrates SHAP based interpretability analysis to improve the transparency and reliability of model decisions.



## 2.Description

### 2.1 Code Overview

This repository contains a complete machine learning pipeline for the early prediction of Alzheimer’s disease, covering data preprocessing, feature selection, model training and hyperparameter optimization, conformal prediction for uncertainty quantification, and SHAP based interpretability analysis. For detailed instructions on code execution, please refer to the ***Code Information*** section.



### 2.2 dataset overview

This study uses a publicly available Alzheimer’s disease dataset comprising comprehensive health information from 2,149 individuals, including 760 patients with AD and 1,389 healthy controls, resulting in a typical class imbalance problem. The dataset encompasses multidimensional features, including demographic characteristics, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, and symptoms. The dataset was obtained from the Kaggle platform. For further details, please refer to the ***Dataset Information*** section.



## 3.Dataset Information:

The dataset used in this study encompasses multidimensional clinical indicators, including demographic characteristics, lifestyle factors, medical history, clinical measurements, and cognitive and functional assessments. Detailed information about the dataset can be accessed through the following link:

\[Data](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

## 4.Code information

This repository contains two core scripts. ***main\_model.py*** serves as the main program and provides a complete implementation of the HOCP framework, including data loading and splitting, Grid CPSO based hybrid hyperparameter optimization for the base models, normalized Mondrian conformal prediction, SHAP based interpretability analysis, and visualization generation. All generated output files are automatically saved in the ***output/*** directory. ***randomforest\_ablation.py*** is the script for the ablation experiments, using Random Forest as the base model and conducting 30 repeated runs to compare the performance of three optimization strategies. It also performs statistical significance tests and saves the resulting files in the ***output\_ablation\_rf2/*** directory.



## 5.Requirements-Any dependencies

numpy==2.3.3

pandas==2.3.2

scikit-learn==1.7.1

imbalanced-learn==0.14.0

xgboost==3.0.1

matplotlib==3.10.6

seaborn==0.13.2

shap==0.51.0

joblib==1.5.2

scipy.stats	==1.16.2

## 6.Methodology

### 6.1 Data Processing

This study employed LASSO regression with L1 regularization for feature selection. The optimal regularization parameter, λ, was determined through 10 fold cross validation, resulting in the selection of 12 key features from the original 34 features. Standardization, LASSO based feature selection, and SMOTE were all performed within the cross validation procedure to ensure that the validation data were not involved in the estimation of preprocessing parameters or the oversampling process, thereby preventing information leakage. SMOTE balances the class distribution by generating synthetic minority class samples through interpolation within the feature space of the minority class.

### 6.2 Modeling and Evaluation

This study selected eight representative base models. These models were not chosen in advance because of any specific advantages, but rather to cover major categories of commonly used classifiers and to demonstrate the model agnostic nature of Grid CPSO and conformal prediction. Accordingly, the proposed framework can be flexibly applied to different classification models. The hyperparameters of all models were optimized using the Grid CPSO hybrid strategy, and the optimal model was selected according to the AUC obtained through 10 fold cross validation.

Performance on the test set was evaluated using accuracy, precision, recall, F1 score, G mean, and AUC. A normalized Mondrian conformal predictor was then constructed using the optimal model and the calibration set. Prediction sets were generated at a 95% confidence level, and their coverage rate and singleton prediction proportion were evaluated. In addition, SHAP analysis was incorporated to quantify feature contributions and identify the key predictors associated with model predictions.



## 7.Computational Infrastructure

python python 3.13.5 Platform: x86\_64-w64-mingw32/x64 (64-bit) Running under: Windows >= 10 x64 (build 26100) The packages loaded:numpy\_2.3.3, pandas\_2.3.2, scikit-learn\_1.7.1,xgboost\_3.0.1, imbalanced-learn\_0.14.0, matplotlib\_3.10.6, seaborn\_0.13.2

