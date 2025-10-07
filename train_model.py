import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score # Importación de CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
import joblib
import warnings
import numpy as np

# Ignorar advertencias de convergencia
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Carga del Dataset y Selección de Características ---
print("Cargando y preparando dataset de Cáncer de Mama...")
data = load_breast_cancer(as_frame=True)
df = data.frame
df['target'] = data.target

# 10 características seleccionadas para simplificar el payload de la API
FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness'
]
TARGET = 'target'

X = df[FEATURES]
y = df[TARGET]

# División de datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Entrenamiento del Modelo (Random Forest) ---
print("Entrenando modelo Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# --- 2.1. Validación Cruzada para Robustez ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_cv_accuracy = np.mean(cv_scores)

print(f"\n--- Validación Cruzada (K=5) ---")
print(f"Scores CV: {cv_scores}")
print(f"Precisión Promedio CV: {mean_cv_accuracy:.4f}")

# Entrenamiento final en el set de entrenamiento
model.fit(X_train, y_train)

# --- 3. Evaluación Detallada en el Set de Prueba (Test Set) ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Evaluación en Conjunto de Prueba ---")
print(f"Precisión (Accuracy): {accuracy:.4f}")
print(f"Recall (Sensibilidad): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# --- 4. Serialización del Modelo y Metadata ---
MODEL_PATH = 'modelo_cancer.joblib'
METADATA = {
    'features': FEATURES,
    'model_accuracy': accuracy,
    'model_recall': recall,
    'model_f1_score': f1,
    'mean_cv_accuracy': mean_cv_accuracy, # Nuevo campo de metadata
    'model_type': 'RandomForestClassifier',
    'version': '2.0.1' # Incremento de versión por mejora de validación
}

# Guardar el modelo
joblib.dump(model, MODEL_PATH)
print(f"\nModelo guardado exitosamente en: {MODEL_PATH}")

# Guardar metadata
joblib.dump(METADATA, 'metadata.joblib')
print("Metadata del modelo guardada, incluyendo métricas clave y CV para trazabilidad.")

# Ejemplo de cómo sería el input JSON para la API
ejemplo_input = dict(zip(FEATURES, X_test.iloc[0].tolist()))
print("\nEjemplo de input JSON necesario para la ruta /predict:")
import json
print(json.dumps(ejemplo_input, indent=2))
