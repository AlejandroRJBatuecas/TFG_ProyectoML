# Parámetros generales para los modelos de ML
data_filename = "ml_subsystem/datasets/dataset_openqasm_qiskit.csv" # Ruta de almacenamiento del dataset
test_data_filename = "ml_subsystem/datasets/file_metrics.csv" # Ruta de almacenamiento del csv con las métricas del fichero a analizar
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
random_state_value = 42 # Valor de la seed de aleatoriedad, para asegurar reproducibilidad
patterns_list = ["p.initialization", "p.superposition", "p.oracle"] # Lista de patrones a detectar
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"] # Métricas de Oráculo eliminadas
min_importance_value = 0.01 # Selecciona características con una importancia superior a este valor
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
cv_value = 3 # Por defecto = 5. Número de particiones realizadas en la validación cruzada. Ponemos 3 ya que es un conjunto de datos pequeño
test_results_num = 10 # Número de registros de prueba mostrados

# Rutas de almacenamiento de modelos
# # K-Nearest Neighbors
knn_classifier_trained_model_path = "ml_subsystem/trained_models/knn_classifier.joblib"