"""
Fichero de configuración que contiene las variables de configuración de los modelos de ML
"""

# Parámetros generales para los modelos de ML
data_filename = "ml_subsystem/datasets/dataset_openqasm_qiskit.csv"
""" Ruta de almacenamiento del dataset """
test_data_filename = "ml_subsystem/datasets/file_metrics.json"
""" Ruta de almacenamiento del json con las métricas del fichero a analizar """
test_set_size = 0.3 # 70% de datos para el conjunto de entrenamiento y 30% para el conjunto de prueba
""" Porcentaje de datos utilizados para el conjunto de prueba """
random_state_value = 42
""" Valor de la seed de aleatoriedad, para asegurar reproducibilidad """
patterns_list = ["p.initialization", "p.superposition", "p.oracle"]
""" Lista de patrones a detectar """
patterns_links = {
    "p.initialization": "https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d/312bc9d3-26c0-40ae-b90b-56effd136c0d", 
    "p.superposition": "https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d/2229a430-fe92-4411-9d72-d10dd1d8da14", 
    "p.oracle": "https://patternatlas.planqk.de/pattern-languages/af7780d5-1f97-4536-8da7-4194b093ab1d/1cc7e9d6-ab37-412e-8afa-604a25de296e"
}
""" Diccionario con las URLs con la documentación de los patrones """
eliminated_columns = ["id", "language", "extension", "author", "name", "path", "circuit"]
""" Columnas eliminadas para realizar la matriz de correlación """
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"]
""" Métricas de Oráculo eliminadas """
min_importance_value = 0.01
""" Selecciona características con una importancia superior a este valor """
min_importance_values = [0.02, 0.02, 0.01]
""" Selecciona características con una importancia superior a este valor para cada etiqueta """
min_correlation_value = 0.5
""" Selecciona características con una correlación superior a este valor """
cv_value = 3 # Establecido en 3, ya que es un conjunto de datos pequeño
""" Por defecto = 5 \n 
Número de particiones realizadas en la validación cruzada """
n_iter_value = 30
""" Por defecto = 10. \n
Número de configuraciones de parámetros que se muestrean para RandomizedSearchCV """
test_results_num = 10
""" Número de registros de prueba mostrados """