import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import cross_val_predict

# Parámetros generales
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
patterns_list = ["p.initialization", "p.superposition", "p.oracle"] # Lista de patrones a detectar
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"] # Métricas de Oráculo eliminadas
min_importance_values = [0.04, 0.04, 0.01] # Selecciona características con una importancia superior a este valor
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
cv_value = 3 # Número de particiones realizadas en la validación cruzada. Por defecto = 5
test_results_num = 5 # Número de registros de prueba mostrados

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Establecer la semilla de aleatoridad
np.random.seed(42)

# Leer el csv con los datos
data = pd.read_csv(Path("../datasets/dataset_openqasm_qiskit.csv"), delimiter=";")

# Limpiar las filas con algún dato nulo
data = data.dropna()

# Elimnar las columnas relacionadas con Oracle
data = data.drop(columns=eliminated_metrics)

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
best_features_scaler = StandardScaler()
scaled_data_values = best_features_scaler.fit_transform(data_values)
best_features_dict = mlutils.get_best_features(data_values, scaled_data_values, data_labels, train_set_values, train_set_labels, min_importance_values)
# # Unir la lista de cada patrón en una única
best_features_list = [feature for pattern in best_features_dict.values() for feature in pattern]
# # Pasar a conjunto para eliminar valores repetidos y después a lista
best_features = list(set(best_features_list))

# Obtener los datos de entrenamiento y prueba con las mejores métricas
best_features_train_set_values = train_set_values[best_features]
best_features_test_set_values = test_set_values[best_features]

# Escalar los atributos
scaler = StandardScaler()
#best_features_scaler = StandardScaler()
# # Ajustar el escalador y tranformar los datos de entrenamiento
train_set_values = scaler.fit_transform(train_set_values)
# # Transformar los datos de prueba
test_set_values = scaler.transform(test_set_values)
# # Lo mismo para los datos mejorados
best_features_train_set_values = best_features_scaler.fit_transform(best_features_train_set_values)
best_features_test_set_values = best_features_scaler.transform(best_features_test_set_values)

# Crear el clasificador multietiqueta
classifier = KNeighborsClassifier()
best_features_classifier = KNeighborsClassifier()

print("\nCaracterísticas de los clasificadores:")
print(classifier)
print(classifier.get_params())

# Transponer las listas colocando los datos en columnas para cada patrón
train_set_labels_np_matrix = np.c_[tuple(train_set_labels[pattern] for pattern in patterns_list)]

# Definir la cuadrícula de hiperparámetros a buscar
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}

# Realizar la búsqueda de hiperparámetros utilizando validación cruzada
grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(train_set_values, train_set_labels_np_matrix)

best_features_grid_search = GridSearchCV(best_features_classifier, param_grid, cv=5)
best_features_grid_search.fit(best_features_train_set_values, train_set_labels_np_matrix)

# Obtener el mejor modelo y sus hiperparámetros
knn_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

best_features_knn_classifier = best_features_grid_search.best_estimator_
best_features_best_params = best_features_grid_search.best_params_

print("\nCaracterísticas de los clasificadores optimizados:")
print(knn_classifier)
print(best_params)

print(best_features_knn_classifier)
print(best_features_best_params)

'''
# Entrenar el clasificador
knn_classifier.fit(train_set_values, train_set_labels_np_matrix)
best_features_knn_classifier.fit(best_features_train_set_values, train_set_labels_np_matrix)
'''

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
predictions_proba = knn_classifier.predict_proba(test_set_values)

# Transponer las listas colocando los datos en columnas para cada patrón
test_set_labels_np_matrix = np.c_[tuple(test_set_labels[pattern] for pattern in patterns_list)]

# # Mostrar las medidas de rendimiento
print("\nRendimiento del modelo con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)

# Realizar predicciones en el conjunto de prueba con las mejores métricas
predictions = best_features_knn_classifier.predict(best_features_test_set_values)
best_features_predictions_proba = best_features_knn_classifier.predict_proba(best_features_test_set_values)

# # Mostrar las medidas de rendimiento
print(f"\nMétricas seleccionadas: {best_features}")
print("\nRendimiento del modelo mejorado con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)

# Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
for i in range(0,test_results_num):
    print(f"\nInstancia {i+1}: {test_set_labels.iloc[i].tolist()}")
    for j in range(len(predictions_proba)):
        print(f"Patrón {patterns_list[j]}: ")
        if len(predictions_proba[j][i]) > 1 and (predictions_proba[j][i][1] > predictions_proba[j][i][0]):
            print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[j][i][1]*100, 2)}%")
        else:
            print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[j][i][0]*100, 2)}%")

        if len(best_features_predictions_proba[j][i]) > 1 and (best_features_predictions_proba[j][i][1] > best_features_predictions_proba[j][i][0]):
            print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[j][i][1]*100, 2)}%")
        else:
            print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[j][i][0]*100, 2)}%")