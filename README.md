# MLOps Cancer Prediction Service | Despliegue Automatizado

Este proyecto representa una solución completa de Machine Learning Operations (MLOps) para el despliegue de un modelo predictivo de diagnóstico de **Cáncer de Mama** (basado en el dataset de Wisconsin). El objetivo es demostrar un flujo de trabajo  que incluye API REST robusta, contenedorización (Docker) y automatización CI/CD, listo para producción.

---

## 1. Stack Tecnológico

| Componente | Herramienta | Nivel de Práctica MLOps |
| :--- | :--- | :--- |
| **Modelo ML** | Random Forest Classifier (scikit-learn) | Modelo robusto con reporte de métricas clave (F1, Recall). |
| **API REST** | Flask + Gunicorn | Servidor WSGI de alto rendimiento para escalabilidad |
| **Contenedorización** | Docker | Entorno reproducible y optimizado con imagen `slim` |
| **Integración Continua** | GitHub Actions | Automatización de tests de endpoints y versionado continuo |


## 2. Estructura del Proyecto

```bash
mlops-cancer-api/
├── app.py                      # API REST con Flask (Validación de entrada y Logging)
├── train_model.py              # Script de entrenamiento (Genera modelo_cancer.joblib)
├── requirements.txt            # Dependencias del proyecto
├── Dockerfile                  # Define el entorno reproducible
└── .github/
    └── workflows/
        └── ci_cd.yml           # Workflow de GitHub Actions (Build, Test y Versionado)
```

## 3. Justificación y Validación del Modelo

El modelo utiliza un **Random Forest Classifier** entrenado con validación cruzada para asegurar la robustez y la interpretabilidad.

| Métrica | Valor | Propósito en el Diagnóstico |
| :--- | :--- | :--- |
| **Precisión Promedio CV** | 0.9613 | **Robustez**: Confirma la estabilidad del modelo a través de diferentes particiones de datos. |
| **Recall (Sensibilidad)** | 1.0000 | **Prioridad Clínica**: Mide la capacidad de identificar todos los casos positivos. El **100%** minimiza los **Falsos Negativos** (el error más costoso). |
| **F1-Score** | 0.9793 | **Equilibrio**: Media armónica entre Precision y Recall, indicando un **alto rendimiento general**. |
| **Accuracy (General)** | 0.9737 | Precisión general de las predicciones. |

Estos resultados, si bien altos, son **característicos de la naturaleza distintiva del dataset** de Cáncer de Mama de Wisconsin. La alta métrica de **Recall** se justifica por la necesidad clínica de **minimizar Falsos Negativos**. La **Validación Cruzada** confirma la **estabilidad y robustez** del modelo en la generalización de los datos.


## 4. Guía de Despliegue Local con Docker

El despliegue se realiza a través de Docker para garantizar la reproducibilidad.

### 4.1. Clonar y Construir la Imagen

El `Dockerfile` ejecuta `train_model.py` durante la construcción, asegurando que el modelo esté entrenado dentro de la imagen.

```bash
# Instalar dependencias
# 1. Clonar el repositorio
git clone <https://github.com/matias-cg/mlops-cancer-api>
cd mlops-cancer-api

# 2. Construir la Imagen (etiqueta inicial v2.0.0)
# NOTA: La construcción está optimizada por capas (instalando dependencias primero)
docker build -t cancer-prediction-api:v2.0.0 .
```

### 4.2. Ejecución y Acceso al Servicio

```bash
# Ejecutar en segundo plano, mapeando el puerto 5000
docker run -d -p 5000:5000 --name cancer_api_prod cancer-prediction-api:v2.0.0
```
El servicio API Gunicorn se inicia en `http://127.0.0.1:5000`.

### 5. Endpoints y Pruebas de la API

La API expone dos rutas:

### 5.1. Estado del Servicio (GET /)

* **Propósito:** Verifica que la API está viva y que el modelo (`joblib`) ha sido cargado correctamente.
* **Comando de prueba:** `curl http://localhost:5000/`

### 5.2. Predicción de Diagnóstico (`POST /predict`)

* **Propósito:** Recibe un JSON y retorna el diagnóstico predicho por el Random Forest. Incluye **validación estricta de entradas y manejo de error 400.**
* **Esquema Requerido (10 Features):**

    * `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`

**Ejemplo de Petición Exitosa (cURL):**

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
        "mean radius": 13.0, "mean texture": 12.0, "mean perimeter": 84.0, 
        "mean area": 520.0, "mean smoothness": 0.08, "worst radius": 15.0, 
        "worst texture": 18.0, "worst perimeter": 95.0, "worst area": 700.0, 
        "worst smoothness": 0.10
     }'

```
## 6. MLOps Avanzado: Flujo CI/CD con GitHub Actions
El flujo definido en `.github/workflows/ci_cd.yml` garantiza la **Integración Continua** del servicio ML.

| Paso | Descripción | Criterio de Sobresaliente |
| :--- | :--- | :--- |
| **Build** | Construcción exitosa de la imagen Docker. | Dockerización reproducible |
| **Test** (GET /) | Verifica la salud del servicio y la carga del modelo. | Pruebas automatizadas de endpoints básicos |
| **Test** (POST /predict) | Ejecuta una predicción exitosa y verifica el código 200 | Pruebas funcionales validadas. |
| **Test** (Error Handling) | Envía datos incompletos y verifica el código **400 Bad Request**, demostrando la robustez de la API. | Manejo de errores validado automáticamente. |
| **Versionado** | La imagen se etiqueta con el **SHA del commit** (`${{ github.sha }}`) después de pasar todas las pruebas. | Flujo  con integración continua y trazabilidad de versiones. |