import os


def train_random_forest(data_file, n_trials=30):
    import pandas as pd
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # -----------------------------
    # Load CSV data
    # -----------------------------
    df = pd.read_csv(data_file, header=None)
    df.columns = ['Range', 'SNR', 'target_class']

    X = df[['Range', 'SNR']]
    y = df['target_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Optuna objective function
    # -----------------------------

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro')
        return scores.mean()
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    #print("\nBest Hyperparameters:")
    #print(study.best_params)

    # -----------------------------
    # Train best model and predict
    # -----------------------------
    rf_best = RandomForestClassifier(**study.best_params, random_state=42)
    rf_best.fit(X_train, y_train)
    y_pred = rf_best.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Small','Medium','Large'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Random Forest Confusion Matrix")
    # plt.show()

    output_dir = "/home/gal/Desktop/Radars/rcs-target-detection/graphs"
    os.makedirs(output_dir, exist_ok=True)


    plt.savefig(os.path.join(output_dir, "rf_confusion_matrix.png"), dpi=300)


    # -----------------------------
    # Prepare dataframe for plotting
    # -----------------------------
    df_test = X_test.copy()
    df_test['true_class'] = y_test
    df_test['pred_class'] = y_pred

    # Return results for comparison
    return {
        'model': rf_best,
        'y_pred': y_pred,
        'y_test': y_test,
        'X_test': X_test,
        'X_train': X_train,
        'y_train': y_train
    }