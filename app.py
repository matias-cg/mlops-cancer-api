import joblib
import numpy as np
from flask import Flask, request, jsonify
import logging
import os
import sys

# --- Configuración de Logging ---
# Configuración profesional de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [API_LOG] - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Inicialización y Carga del Modelo ---
MODEL_PATH = 'modelo_cancer.joblib'
METADATA_PATH = 'metadata.joblib'

app = Flask(__name__)
model = None
model_metadata = None
REQUIRED_FEATURES = None

def load_resources():
    """Carga el modelo y la metadata en memoria al iniciar la aplicación."""
    global model, model_metadata, REQUIRED_FEATURES
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            logger.error(f"Archivos de modelo o metadata no encontrados en /app. Asegúrate de ejecutar 'train_model.py' en el entorno de build.")
            # Si los archivos no existen, el servicio debe fallar o funcionar en estado degradado.
            return 
        
        model = joblib.load(MODEL_PATH)
        model_metadata = joblib.load(METADATA_PATH)
        REQUIRED_FEATURES = model_metadata.get('features')
        logger.info(f"Modelo ({model_metadata.get('model_type', 'Desconocido')}) cargado exitosamente. Versión: {model_metadata.get('version', 'unknown')}")
    except Exception as e:
        logger.error(f"FATAL: Error al cargar recursos. El servicio no puede operar. Detalle: {e}", exc_info=True)
        pass 

# Se llama al inicio del script para cargar el modelo
load_resources()

# --- Rutas de la API ---

@app.route('/', methods=['GET'])
def home():
    """Ruta para verificar el estado del servicio y la versión del modelo."""
    if model is None:
        return jsonify({
            "status": "error",
            "service": "MLOps Cancer Prediction API",
            "message": "Modelo no inicializado. El servicio está en estado degradado (500)."
        }), 500
        
    return jsonify({
        "service": "MLOps Cancer Prediction API",
        "status": "ok",
        "model_type": model_metadata.get('model_type', 'unknown'),
        "model_version": model_metadata.get('version', 'unknown'),
        "model_accuracy": f"{model_metadata.get('model_accuracy', 0.0) * 100:.2f}%",
        "instructions": "Use POST /predict para obtener un diagnóstico. Consulte el README para el formato de JSON."
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para recibir datos, validarlos y retornar una predicción."""
    if model is None:
        logger.warning("Intento de predicción fallido: Modelo no cargado.")
        return jsonify({"error": "El modelo no está disponible. Revisa el estado del servicio GET /"}), 503

    try:
        # 1. Obtener los datos del JSON
        data = request.get_json(silent=True)
        if not data:
            logger.warning("Solicitud con JSON inválido o nulo.")
            return jsonify({"error": "JSON de entrada inválido o vacío. Asegure Content-Type: application/json."}), 400

        # 2. Validación y Mapeo (Simulación de Pydantic/Schemas)
        input_values = []
        missing_features = [f for f in REQUIRED_FEATURES if f not in data]
        non_numeric_features = []

        if missing_features:
            error_msg = f"Error 400: Faltan las siguientes características requeridas ({len(missing_features)}/{len(REQUIRED_FEATURES)}): {', '.join(missing_features)}"
            logger.warning(f"Error de validación: {error_msg}")
            return jsonify({
                "error": error_msg,
                "required_schema": REQUIRED_FEATURES
            }), 400

        for feature in REQUIRED_FEATURES:
            try:
                # Intento de conversión robusta a float
                value = float(data[feature])
                input_values.append(value)
            except (ValueError, TypeError):
                non_numeric_features.append(feature)
        
        if non_numeric_features:
            error_msg = f"Error 400: Las características deben ser numéricas. Revise: {', '.join(non_numeric_features)}"
            logger.warning(f"Error de validación: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # 3. Preparar los datos y realizar la predicción
        features_array = np.array([input_values])
        prediction = model.predict(features_array)[0]
        
        # 4. Formatear la respuesta (Interpretación)
        # 0: Benigno (No Cáncer), 1: Maligno (Cáncer)
        prediction_label = "Maligno" if prediction == 1 else "Benigno"
        
        logger.info(f"Predicción exitosa. Diagnóstico: {prediction_label}. ID_peticion: {request.environ.get('REMOTE_ADDR')}")
        
        return jsonify({
            "prediction_code": int(prediction), 
            "result_label": prediction_label,
            "message": "Diagnóstico predictivo completado.",
            "model_version": model_metadata.get('version', 'unknown')
        }), 200

    except Exception as e:
        # 5. Manejo de errores genérico (500)
        logger.error(f"ERROR 500: Falla crítica durante el procesamiento de la solicitud. Detalle: {e}", exc_info=True)
        return jsonify({
            "error": "Error interno del servidor al procesar la solicitud.",
            "details": "Verificar logs para más información."
        }), 500

if __name__ == '__main__':
    # Usar gunicorn es la mejor práctica, incluso para desarrollo local robusto
    logger.info("Iniciando API Flask/Gunicorn en modo desarrollo.")
    # Usamos gunicorn programáticamente si no estamos en Docker, aunque en Docker se usa el CMD
    # Para simplicidad local:
    # app.run(debug=True, host='0.0.0.0', port=5000) 
    # Mejor: Instruir al usuario a usar gunicorn o docker run
    logger.info("Por favor, ejecute con 'gunicorn --bind 0.0.0.0:5000 app:app' o use Docker.")
