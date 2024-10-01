"""
Fichero de configuración que contiene las rutas donde están almacenados los modelos de ML entrenados
"""

trained_models = {
    "Multi-label KNN classifier": "knn_classifier",
    "Individual KNN classifier": "knn_one_vs_rest_classifier",
    "Individual Random Forest classifier": "random_forest_classifier"
}

# Directorio de almacenamiento de los modelos
trained_model_base_path = "ml_subsystem/trained_models/"

# Extensión del fichero de almacenamiento de los modelos
trained_model_extension = ".joblib"