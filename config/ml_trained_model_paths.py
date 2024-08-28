"""
Fichero de configuración que contiene las rutas donde están almacenados los modelos de ML entrenados
"""

trained_models = {
    "Clasificador KNN multietiqueta": "knn_classifier",
    "Clasificador KNN individual": "knn_one_vs_rest_classifier",
    "Clasificador Random Forest individual": "random_forest_classifier"
}

# Directorio de almacenamiento de los modelos
trained_model_base_path = "ml_subsystem/trained_models/"

# Extensión del fichero de almacenamiento de los modelos
trained_model_extension = ".joblib"