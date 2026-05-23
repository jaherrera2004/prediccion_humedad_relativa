import json
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("data/weatherHistory_clean.csv", parse_dates=["formatted_date"])

# Discretizar humedad en 3 categorías
def clasificar_humedad(h):
    if h < 0.40:
        return "Baja"
    elif h < 0.70:
        return "Media"
    else:
        return "Alta"

df["humedad_cat"] = df["humidity"].apply(clasificar_humedad)

# Extraer features temporales
df["mes"]        = df["formatted_date"].dt.month
df["hora"]       = df["formatted_date"].dt.hour
df["dia_semana"] = df["formatted_date"].dt.dayofweek

# Definir features y variable objetivo
features = [
    "temperature", "apparent_temperature", "wind_speed",
    "wind_bearing", "visibility", "pressure",
    "mes", "hora", "dia_semana",
]

X = df[features]
y = df["humedad_cat"]

# Split 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline
baseline = DecisionTreeClassifier(random_state=42, class_weight="balanced")
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
print(f"Baseline — accuracy: {accuracy_score(y_test, y_pred_base):.4f} | f1 macro: {f1_score(y_test, y_pred_base, average='macro'):.4f}")

# Ajuste de hiperparámetros
param_grid = {
    "max_depth":        [3, 5, 10, 15, None],
    "min_samples_split":[2, 10, 50],
    "min_samples_leaf": [1, 5, 20],
    "criterion":        ["gini", "entropy"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
grid.fit(X_train, y_train)
print(f"Mejores parámetros: {grid.best_params_}")

# Reentrenar con mejores parámetros
best = grid.best_estimator_
y_pred_best = best.predict(X_test)
print(f"Optimizado — accuracy: {accuracy_score(y_test, y_pred_best):.4f} | f1 macro: {f1_score(y_test, y_pred_best, average='macro'):.4f}")

# Validación cruzada (5-fold)
cv_scores = cross_val_score(best, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"CV f1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Guardar métricas en JSON para el backend
classes = sorted(y.unique())

metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred_best), 4),
    "f1_macro":  round(f1_score(y_test, y_pred_best, average="macro"), 4),
    "cv_f1_mean": round(cv_scores.mean(), 4),
    "cv_f1_std":  round(cv_scores.std(), 4),
    "per_class": {
        cls: {
            "precision": round(precision_score(y_test, y_pred_best, labels=[cls], average="macro", zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred_best, labels=[cls], average="macro", zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred_best, labels=[cls], average="macro", zero_division=0), 4),
        }
        for cls in classes
    },
    "confusion_matrix": confusion_matrix(y_test, y_pred_best, labels=classes).tolist(),
    "confusion_matrix_labels": classes,
    "feature_importances": dict(zip(features, [round(v, 4) for v in best.feature_importances_])),
    "best_params": grid.best_params_,
}

os.makedirs("models", exist_ok=True)
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Métricas guardadas en models/metrics.json")

# Exportar modelo, encoder y lista de features
joblib.dump(best, "models/model.pkl")

with open("models/features.json", "w") as f:
    json.dump(features, f, indent=2)

print("Modelo exportado en models/")
