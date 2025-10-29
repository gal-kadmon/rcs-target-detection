import os


def train_mlp(data_file, n_trials=20):
    """
    Train a small MLP classifier with Optuna on scaled features.
    
    Parameters:
        data_file (str): Path to CSV file with columns [Range, SNR, target_class]
        n_trials (int): Number of Optuna trials
    
    Returns:
        dict: {
            'model': trained MLP,
            'y_pred': predictions on test set,
            'y_test': true labels,
            'X_test': X test set,
            'X_train': X train set,
            'y_train': y train set
        }
    """
    import pandas as pd
    import optuna
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Load CSV data
    df = pd.read_csv(data_file, header=None)
    df.columns = ['Range', 'SNR', 'target_class']

    X = df[['Range', 'SNR']]
    y = df['target_class']

    # -----------------------------
    # Scale features
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Optuna objective
    # -----------------------------
    def objective(trial):
        hidden_layer_1 = trial.suggest_int('hidden_layer_1', 3, 15)
        hidden_layer_2 = trial.suggest_int('hidden_layer_2', 0, 10)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 0.01, log=True)
        
        if hidden_layer_2 > 0:
            hidden_layers = (hidden_layer_1, hidden_layer_2)
        else:
            hidden_layers = (hidden_layer_1,)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=2000,
            random_state=42
        )

        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='f1_macro')
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    #print("\nBest Hyperparameters (MLP):")
    #print(study.best_params)

    # -----------------------------
    # Train best model
    # -----------------------------
    hl1 = study.best_params['hidden_layer_1']
    hl2 = study.best_params['hidden_layer_2']
    if hl2 > 0:
        hidden_layers = (hl1, hl2)
    else:
        hidden_layers = (hl1,)

    mlp_best = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=study.best_params['activation'],
        alpha=study.best_params['alpha'],
        learning_rate_init=study.best_params['learning_rate_init'],
        max_iter=1000,
        random_state=42
    )

    mlp_best.fit(X_train, y_train)
    y_pred = mlp_best.predict(X_test)

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("\nMLP Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Small','Medium','Large'])
    disp.plot(cmap=plt.cm.Blues)
    # plt.show()

    output_dir = "/home/gal/Desktop/Radars/rcs-target-detection/graphs"
    os.makedirs(output_dir, exist_ok=True)


    plt.savefig(os.path.join(output_dir, "mlp_confusion_matrix.png"), dpi=300)

    # -----------------------------
    # Return results for comparison
    # -----------------------------
    return {
        'model': mlp_best,
        'y_pred': y_pred,
        'y_test': y_test,
        'X_test': X_test,
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler
    }
