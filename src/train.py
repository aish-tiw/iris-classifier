# src/train.py

import argparse
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import joblib


parser = argparse.ArgumentParser(description="Notebook-style Iris training (DT GridSearchCV or KNN).")
parser.add_argument("--model", choices=["dt", "knn"], default="dt", help="Choose model: dt or knn.")
parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test split.")
parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
parser.add_argument("--k", type=int, default=5, help="K for KNN (when --model knn).")
parser.add_argument("--outputs-dir", type=str, default="outputs", help="Where to save results.")
parser.add_argument("--save-extra-plots", action="store_true", help="Also save countplot, pairplot, feature/tree plots.")
args = parser.parse_args()

# Setup paths 
outputs_dir = Path(args.outputs_dir)
outputs_dir.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

# Load data
iris = load_iris(as_frame=True)
df = iris.frame.copy()
X = iris.data
y = iris.target
target_names = iris.target_names

# EDA plots
df["species"] = df["target"].apply(lambda i: iris.target_names[i])

if args.save_extra_plots:
    # Countplot
    plt.figure()
    sns.countplot(x="species", data=df)
    plt.title("Distribution of Species")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outputs_dir / "countplot.png", dpi=150)
    plt.close()

    # Pairplot 
    pair = sns.pairplot(df, hue="species")
    pair.fig.suptitle("Pair Plot of Features by Species", y=1.02)
    pair.savefig(outputs_dir / "pairplot.png", dpi=150)
    plt.close("all")

# Train / Test split and Scaling 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Shapes -> X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, "
      f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Model training
if args.model == "dt":
    # Decision Tree with GridSearchCV 
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    dt = DecisionTreeClassifier(random_state=args.random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)

    
    grid_search.fit(X_train_scaled, y_train)

    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test_scaled)

elif args.model == "knn":
    knn = KNeighborsClassifier(n_neighbors=args.k)
    knn.fit(X_train_scaled, y_train)
    model = knn
    y_pred = model.predict(X_test_scaled)

else:
    raise ValueError("Unknown model choice.")

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
print(f"\nModel: {args.model.upper()} | Accuracy: {acc:.4f}")

# Confusion matrix 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
fig.tight_layout()
cm_path = outputs_dir / "confusion_matrix.png"
fig.savefig(cm_path, dpi=150)
plt.close(fig)

# Save trained model
if args.model == "dt":
    model_path = outputs_dir / "dt_model.joblib"
else:
    model_path = outputs_dir / "knn_model.joblib"

scaler_path = outputs_dir / "scaler.joblib"
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Saved confusion matrix to: {cm_path}")
print(f"Saved model to: {model_path}")
print(f"Saved scaler to: {scaler_path}")

# Feature importances + Tree plot
if args.model == "dt" and args.save_extra_plots:
    # Feature importances
    feature_imp = getattr(model, "feature_importances_", None)
    if feature_imp is not None:
        feature_names = X.columns
        feature_imp_df = pd.DataFrame({"feature": feature_names, "importance": feature_imp}).sort_values(
            "importance", ascending=False
        )
        plt.figure(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=feature_imp_df)
        plt.title("Feature Importances from Decision Tree")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(outputs_dir / "feature_importances.png", dpi=150)
        plt.close()

    # Tree structure
    plt.figure(figsize=(24, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=list(target_names),
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(outputs_dir / "tree.png", dpi=150)
    plt.close()


