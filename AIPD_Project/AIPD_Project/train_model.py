import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "predictive_maintenance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Remove unnecessary columns
data.drop(['UDI', 'Product ID', 'Failure Type'], axis=1, inplace=True)

# Encode Machine Type
encoder = LabelEncoder()
data['Type'] = encoder.fit_transform(data['Type'])

# Features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and compare multiple models
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    ),
}

results = []
best_model_name = None
best_model = None
best_selection_score = float("-inf")
max_allowed_gap = 0.15

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_gap = train_f1 - test_f1
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_f1_mean = cv_scores.mean()
    cv_f1_std = cv_scores.std()

    # Penalize models that overfit (large train-test F1 gap).
    selection_score = cv_f1_mean - max(0.0, f1_gap) * 0.25

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": test_f1,
        "Train_F1": train_f1,
        "F1_Gap": f1_gap,
        "CV_F1_Mean": cv_f1_mean,
        "CV_F1_Std": cv_f1_std,
        "Selection_Score": selection_score,
    }
    results.append(metrics)

    if (f1_gap <= max_allowed_gap) and (selection_score > best_selection_score):
        best_selection_score = selection_score
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(results).sort_values(by="Selection_Score", ascending=False)

# Fallback: if all models exceed the overfitting threshold, pick highest selection score.
if best_model is None:
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model = models[best_model_name]

# Save model and encoder
joblib.dump(best_model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

# Save comparison report
comparison_path = BASE_DIR / "model_comparison.csv"
results_df.to_csv(comparison_path, index=False)

print("✅ Model trained successfully")
print(f"📦 Model saved as: {MODEL_PATH.name}")
print(f"🔠 Encoder saved as: {ENCODER_PATH.name}")
print(f"🏆 Best model selected: {best_model_name}")
print("\n📊 Model Comparison:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\n📝 Comparison report saved as: {comparison_path.name}")