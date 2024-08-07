"""
Fichero de configuración que contiene las rutas donde están almacenados los modelos de ML entrenados
"""

trained_models = {
    "Clasificador KNN multietiqueta": "knn_classifier",
    "Clasificador KNN individual": "knn_one_vs_rest_classifier",
    "Clasificador Random Forest individual": "random_forest_classifier"
}

# Rutas de almacenamiento de modelos
knn_classifier_trained_model_path = "ml_subsystem/trained_models/knn_classifier.joblib"
""" Ruta de almacenamiento del modelo K-Nearest Neighbors """
knn_one_vs_rest_classifier_trained_model_path = "ml_subsystem/trained_models/knn_one_vs_rest_classifier.joblib"
""" Ruta de almacenamiento del modelo K-Nearest Neighbors binario para cada etiqueta """
random_forest_classifier_trained_model_path = "ml_subsystem/trained_models/random_forest_classifier.joblib"
""" Ruta de almacenamiento del modelo Random Forest Classifier """