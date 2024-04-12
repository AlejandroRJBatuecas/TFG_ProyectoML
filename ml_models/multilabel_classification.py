import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import cross_val_predict

# Parámetros generales
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
patterns_list = ["p.initialization", "p.superposition", "p.oracle", "p.entanglement"] # Lista de patrones a detectar
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
min_importance_value = 0.04 # Selecciona características con una importancia superior a este valor
cv_value = 3 # Número de particiones realizadas en la validación cruzada. Por defecto = 5

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Leer el csv con los datos
data = pd.read_csv(Path("../datasets/dataset_openqasm_qiskit.csv"), delimiter=";")
test_data = pd.read_csv(Path("../datasets/prueba_cinco_dataset_openqasm_qiskit.csv"), delimiter=";")

# Limpiar las filas con algún dato nulo
data = data.dropna()

# Separar datos y etiquetas
data_values = data.select_dtypes(include=[np.number])
data_labels = data[patterns_list]

# Obtener el conjunto de entrenamiento y de prueba
train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(data_values, data_labels, test_size=test_set_size, stratify=data_labels)

# Mostrar la estructura de los datos
mlutils.show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
#mlutils.create_data_histogram(data)

# Ver la correlación entre los datos
data_vars = data.drop(["id", "language", "extension", "author", "name", "path", "circuit"], axis=1)
# # Obtener la matriz de correlación
mlutils.get_correlation_matrix(data_vars, min_correlation_value, patterns_list)
# # Obtener las mejores características
best_features = mlutils.get_best_features(data_values, data_labels, train_set_values, train_set_labels, min_importance_value)

# Obtener los datos de entrenamiento y prueba con las mejores métricas
best_features_train_set_values = train_set_values[best_features]
best_features_test_set_values = test_set_values[best_features]

# Escalar los atributos
scaler = StandardScaler()
best_features_scaler = StandardScaler()
# # Ajustar el escalador y tranformar los datos de entrenamiento
train_set_values = scaler.fit_transform(train_set_values)
# # Transformar los datos de prueba
test_set_values = scaler.transform(test_set_values)
# # Lo mismo para los datos mejorados
best_features_train_set_values = best_features_scaler.fit_transform(best_features_train_set_values)
best_features_test_set_values = best_features_scaler.transform(best_features_test_set_values)

# Crear el clasificador multietiqueta
knn_classifier = KNeighborsClassifier()
best_features_knn_classifier = KNeighborsClassifier()

# Transponer las listas colocando los datos en columnas para cada patrón
train_set_labels_np_matrix = np.c_[tuple(train_set_labels[pattern] for pattern in patterns_list)]

# Entrenar el clasificador
knn_classifier.fit(train_set_values, train_set_labels_np_matrix)
best_features_knn_classifier.fit(best_features_train_set_values, train_set_labels_np_matrix)

# Evaluar el rendimiento del modelo
# # Calcular las predicciones del clasificador mediante evaluación cruzada
predictions = cross_val_predict(knn_classifier, train_set_values, train_set_labels_np_matrix, cv=cv_value)

# # Mostrar las medidas de rendimiento
print("\nRendimiento del modelo entrenado")
mlutils.model_performance_data(train_set_labels_np_matrix, predictions, patterns_list)

# Evaluar el rendimiento del modelo con las mejores métricas
# # Calcular las predicciones del clasificador mediante evaluación cruzada
predictions = cross_val_predict(best_features_knn_classifier, best_features_train_set_values, train_set_labels_np_matrix, cv=cv_value)

# # Mostrar las medidas de rendimiento
print(f"\nMétricas seleccionadas: {best_features}")
print("Rendimiento del modelo mejorado")
mlutils.model_performance_data(train_set_labels_np_matrix, predictions, patterns_list)

# Realizar predicciones en el conjunto de prueba
predictions = knn_classifier.predict(test_set_values)

# Transponer las listas colocando los datos en columnas para cada patrón
test_set_labels_np_matrix = np.c_[tuple(test_set_labels[pattern] for pattern in patterns_list)]

# # Mostrar las medidas de rendimiento
print("\nRendimiento del modelo con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)

# Realizar predicciones en el conjunto de prueba con las mejores métricas
predictions = best_features_knn_classifier.predict(best_features_test_set_values)

# # Mostrar las medidas de rendimiento
print(f"\nMétricas seleccionadas: {best_features}")
print("\nRendimiento del modelo mejorado con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)

'''
# Preparar los datos de prueba
# # Obtener los datos numéricos
test_set_data_values = test_data.select_dtypes(include=[np.number])
# # Obtener los datos de prueba con las mejores métricas
best_features_test_set_data_values = test_data[best_features]

# Escalar los datos de prueba
test_set_data_values = scaler.transform(test_set_data_values)
best_features_test_set_data_values = best_features_scaler.transform(best_features_test_set_data_values)

# Obtener predicciones y probabilidades de los datos de prueba con el modelo entrenado
test_data_pred = knn_classifier.predict(test_set_data_values)
predictions_proba = knn_classifier.predict_proba(test_set_data_values)

# Realizar predicciones con los datos de prueba
best_features_test_data_pred = best_features_knn_classifier.predict(best_features_test_set_data_values)
best_features_predictions_proba = best_features_knn_classifier.predict_proba(best_features_test_set_data_values)

# Imprimir los porcentajes de predicción
for i in range(len(predictions_proba[0])):
    print(f"\nInstancia {i+1}:")
    for j in range(len(predictions_proba)):
        if len(predictions_proba[j][i]) > 1 and (predictions_proba[j][i][1] > predictions_proba[j][i][0]):
            print(f" Cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][1]*100, 2)}%")
        else:
            print(f" No cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][0]*100, 2)}%")

        if len(best_features_predictions_proba[j][i]) > 1 and (best_features_predictions_proba[j][i][1] > best_features_predictions_proba[j][i][0]):
            print(f" Cumple el patrón {patterns_list[j]} en un {round(best_features_predictions_proba[j][i][1]*100, 2)}%")
        else:
            print(f" No cumple el patrón {patterns_list[j]} en un {round(best_features_predictions_proba[j][i][0]*100, 2)}%")
'''