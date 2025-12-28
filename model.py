import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# 1. Load Dataset
df = pd.read_csv("students_dropout.csv")
target_col = "Target"

# Binary target
df["Target_binary"] = df[target_col].apply(
    lambda x: 1 if x == "Dropout" else 0
)

print("Target distribution:")
print(df[target_col].value_counts())

# 2. Separate features
categorical_features = []
numeric_features = []

exclude = {target_col, "Target_binary"}

for col in df.columns:
    if col in exclude:
        continue
    if df[col].dtype == "object" and df[col].nunique() <= 20:
        categorical_features.append(col)
    else:
        numeric_features.append(col)

print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)

# 3. Preprocessing
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X = df.drop(columns=[target_col, "Target_binary"])
y_multi = df[target_col]
y_binary = df["Target_binary"]

X_pre = preprocessor.fit_transform(X)

print("Total features after preprocessing:", X_pre.shape[1])

# 4. PCA + KMeans (EDA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pre)

for k in [2, 3]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pre)
    sil = kmeans.inertia_
    print(f"K={k}")

    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=5)
    plt.title(f"PCA Clusters (K={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

# 5. Multiclass Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    X_pre, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

clf_multi = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial"
)

clf_multi.fit(X_train, y_train)
pred_multi = clf_multi.predict(X_test)

print("\n=== MULTICLASS LOGISTIC REGRESSION ===")
print(classification_report(y_test, pred_multi))
print("Accuracy:", accuracy_score(y_test, pred_multi))

ConfusionMatrixDisplay.from_predictions(y_test, pred_multi)
plt.title("Confusion Matrix — Multiclass Logistic Regression")
plt.show()

# 6. Multiclass Decision Tree
tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced")
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

print("\n=== MULTICLASS DECISION TREE ===")
print(classification_report(y_test, pred_tree))
print("Accuracy:", accuracy_score(y_test, pred_tree))

ConfusionMatrixDisplay.from_predictions(y_test, pred_tree)
plt.title("Confusion Matrix — Decision Tree")
plt.show()

# 7. Binary Logistic Regression (MAIN MODEL)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_pre, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

clf_bin = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf_bin.fit(Xb_train, yb_train)
pred_bin = clf_bin.predict(Xb_test)
proba = clf_bin.predict_proba(Xb_test)[:, 1]

print("\n=== BINARY LOGISTIC REGRESSION ===")
print(classification_report(yb_test, pred_bin))
print("Accuracy:", accuracy_score(yb_test, pred_bin))

ConfusionMatrixDisplay.from_predictions(yb_test, pred_bin)
plt.title("Confusion Matrix — Binary Logistic Regression")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(yb_test, proba)
auc = roc_auc_score(yb_test, proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Binary Logistic Regression")
plt.legend()
plt.show()
