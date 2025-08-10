import numpy as np
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(self.save_dir, filename))
plt.close()


# Main execution process
if __name__ == "__main__":
    # Initialize components
    preparer = DataPreparer(test_size=0.2, calib_size=0.5, random_state=42)
    trainer = ModelTrainer(random_state=42)
    visualizer = ResultVisualizer()
    conformal = AdvancedConformalPredictor(confidence_level=0.95)

    # Data preparation
    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test = preparer.load_and_split(
            "COUNT_SIS_selected_features.csv.csv"
        )
        print(f"Data loaded successfully! Training set: {X_train.shape}, Calibration set: {X_cal.shape}, Test set: {X_test.shape}")
    except FileNotFoundError:
        print("Error: Data file not found! Please check the file path")
        exit()

    # Model training and evaluation
    print("\nPerforming cross-validation...")
    cv_results = trainer.cross_validate(X_train, y_train)
    print("Cross-validation results:")
    print(cv_results.describe().loc[['mean', 'std']].T)
    visualizer.plot_metrics(cv_results)

    # Train final model
    print("\nTraining final model...")
    final_model = trainer.train_final_model(X_train, y_train)

    # Conformal prediction
    print("\nPerforming split conformal prediction...")
    test_sets, q_hat, coverage_ci = conformal.predict_with_confidence(
        final_model, X_cal, y_cal, X_test
    )

    # Visualize error distribution
    conformal.plot_error_distribution(final_model, X_cal, y_cal)

    # Results analysis
    coverage = np.mean([y_test[i] in test_sets[i] for i in range(len(y_test))])
    avg_set_size = np.mean([len(s) for s in test_sets])

    # Regular evaluation
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    print("\n=== Model Evaluation Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    print(f"G-Mean: {geometric_mean_score(y_test, y_pred):.4f}")

    print("\n=== Split Conformal Prediction Results ===")
    print(f"Actual coverage: {coverage:.4f} (Target ≥ {1 - conformal.alpha:.0%})")
    print(f"Coverage 95% Confidence Interval: [{coverage_ci[0]:.4f}, {coverage_ci[1]:.4f}]")
    print(f"Average prediction set size: {avg_set_size:.4f}")

    # Testing coverage and set size at different confidence levels
    print("\nTesting coverage and prediction set size at different confidence levels...")
    confidence_levels = np.linspace(0.5, 0.99, 10)
    coverage_rates = []
    avg_set_sizes = []

    for cl in confidence_levels:
        temp_conformal = AdvancedConformalPredictor(confidence_level=cl)
        temp_sets, _, _ = temp_conformal.predict_with_confidence(final_model, X_cal, y_cal, X_test)
        coverage_rates.append(np.mean([y_test[i] in temp_sets[i] for i in range(len(y_test))]))
        avg_set_sizes.append(np.mean([len(s) for s in temp_sets]))

    # Visualization
    visualizer.plot_roc_pr(y_test, y_proba)
    visualizer.plot_coverage_vs_set_size(confidence_levels, coverage_rates, avg_set_sizes)
    print("\nAll visualization results have been saved to the results/ directory")
