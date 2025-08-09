software: python
python 3.13
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows >= 10 x64 (build 26100)
The packages loaded:numpy_1.19.0, pandas_1.1.0, scikit-learn_0.24.0,xgboost_1.3.0, imbalanced-learn_0.8.0, matplotlib_3.3.0, seaborn_0.11.0

====>>file "data">>===========================================================================================

The file "code" contains four main classes: DataPreparer, ModelTrainer, AdvancedConformalPredictor, and ResultVisualizer. DataPreparer handles data loading and preprocessing, ModelTrainer manages model training, AdvancedConformalPredictor performs conformal prediction, and ResultVisualizer generates visual outputs.

DataPreparer: to load and preprocess data with stratified splitting and SMOTE oversampling
ModelTrainer: to train and evaluate XGBoost models with cross-validation
AdvancedConformalPredictor: to generate prediction sets with statistical guarantees using conformal prediction
ResultVisualizer: to create diagnostic plots and performance visualizations

Key methods:

1. DataPreparer.load_and_split():
   • Performs stratified train-test split
   • Applies SMOTE to balance training data
   • Splits into training and calibration sets

2. ModelTrainer.cross_validate():
   • Executes stratified k-fold cross-validation
   • Returns accuracy, F1, ROC AUC, G-mean and average precision

3. AdvancedConformalPredictor.predict_with_confidence():
   • Computes nonconformity scores
   • Generates prediction sets with guaranteed coverage
   • Calculates bootstrap confidence intervals

4. ResultVisualizer visualization methods:
   • plot_metrics(): Cross-validation metrics boxplot
   • plot_roc_pr(): ROC and Precision-Recall curves
   • plot_coverage_vs_set_size(): Coverage analysis
   • plot_error_distribution(): Nonconformity score visualization

