from sklearn.metrics import accuracy_score
import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import cross_val_predict

# Parámetros generales
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
patterns_list = ["p.initialization", "p.superposition", "p.oracle"] # Lista de patrones a detectar
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"] # Métricas de Oráculo eliminadas
min_importance_values = [0.005, 0.01, 0.001] # Selecciona características con una importancia superior a este valor
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
#max_importance_feature_num = 4 # Número máximo de características seleccionadas para cada patrón
cv_value = 3 # Número de particiones realizadas en la validación cruzada. Por defecto = 5
test_results_num = 10 # Número de registros de prueba mostrados

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
print(data_labels)
# Obtener el conjunto de entrenamiento y de prueba
train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(data_values, data_labels, test_size=test_set_size, stratify=data_labels)

# Mostrar la estructura de los datos
mlutils.show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
#mlutils.create_data_histogram(data)

# Ver la correlación entre los datos
data_vars = data.drop(["id", "language", "extension", "author", "name", "path", "circuit"], axis=1)
# # Obtener la matriz de correlación
mlutils.get_correlation_matrix(data_vars, min_correlation_value, patterns_list)

best_features_scaler = StandardScaler()
scaled_data_values = best_features_scaler.fit_transform(data_values)
# # Obtener las mejores características
best_features = mlutils.get_best_features(data_values, scaled_data_values, data_labels, train_set_values, train_set_labels, min_importance_values)

# Escalar los atributos
scaler = StandardScaler()

# Crear la lista de clasificadores
classifiers = []
best_features_classifiers = []

# Crear las listas de predicciones
model_predictions = []
model_best_features_predictions = []

predictions = []
predictions_proba = []
best_features_predictions = []
best_features_predictions_proba = []

# Definir la cuadrícula de hiperparámetros a ajustar
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], # Parámetro de regularización
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']  # Algoritmo de optimización
}

# Para cada patrón, crear y entrenar un clasificador
for pattern in patterns_list:
    # Obtener los datos de entrenamiento y prueba con las mejores métricas del patrón
    best_features_train_set_values = train_set_values[best_features[pattern]]
    best_features_test_set_values = test_set_values[best_features[pattern]]
    # Ajustar el escalador y tranformar los datos de entrenamiento
    scaled_train_set_values = scaler.fit_transform(train_set_values)
    # Transformar los datos de prueba
    scaled_test_set_values = scaler.transform(test_set_values)
    # Ajustar el escalador y tranformar los datos de entrenamiento
    scaled_best_features_train_set_values = best_features_scaler.fit_transform(best_features_train_set_values)
    # Transformar los datos de prueba
    scaled_best_features_test_set_values = best_features_scaler.transform(best_features_test_set_values)
    # Obtener las etiquetas del patrón
    pattern_train_labels = train_set_labels[pattern]
    # Crear el clasificador
    #regressor = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    regressor = LogisticRegression(max_iter=10000)
    #best_features_regressor = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    best_features_regressor = LogisticRegression(max_iter=10000)

    print("\nCaracterísticas de los clasificadores:")
    print(regressor)
    print(regressor.get_params())
    # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
    grid_search = GridSearchCV(regressor, param_grid, cv=5)
    grid_search.fit(scaled_train_set_values, pattern_train_labels)

    best_features_grid_search = GridSearchCV(best_features_regressor, param_grid, cv=5)
    best_features_grid_search.fit(scaled_best_features_train_set_values, pattern_train_labels)

    # Obtener el mejor modelo y sus hiperparámetros
    logistic_regressor = grid_search.best_estimator_
    best_params = grid_search.best_params_

    best_features_logistic_regressor = best_features_grid_search.best_estimator_
    best_features_best_params = best_features_grid_search.best_params_

    print(f"\nCaracterísticas de los clasificadores optimizados - {pattern}")
    print(logistic_regressor)
    print(best_params)

    print(best_features_logistic_regressor)
    print(best_features_best_params)
    # Entrenar el clasificador
    #logistic_regressor.fit(scaled_train_set_values, pattern_train_labels)
    #best_features_logistic_regressor.fit(scaled_best_features_train_set_values, pattern_train_labels)
    # Agregar el clasificador entrenado a la lista de clasificadores
    classifiers.append(logistic_regressor)
    best_features_classifiers.append(best_features_logistic_regressor)
    # Realizar las predicciones con el clasificador
    predictions.append(logistic_regressor.predict(scaled_test_set_values))
    predictions_proba.append(logistic_regressor.predict_proba(scaled_test_set_values))
    best_features_predictions.append(best_features_logistic_regressor.predict(scaled_best_features_test_set_values))
    best_features_predictions_proba.append(best_features_logistic_regressor.predict_proba(scaled_best_features_test_set_values))

    # Evaluar el rendimiento del modelo
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    cross_val_predictions = cross_val_predict(logistic_regressor, scaled_train_set_values, pattern_train_labels, cv=cv_value)
    model_predictions.append(cross_val_predictions)

    # Evaluar el rendimiento del modelo con las mejores métricas
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    cross_val_predictions = cross_val_predict(best_features_logistic_regressor, scaled_best_features_train_set_values, pattern_train_labels, cv=cv_value)
    model_best_features_predictions.append(cross_val_predictions)

# Convertir las predicciones en el formato esperado
model_predictions = np.array(model_predictions).T
model_best_features_predictions = np.array(model_best_features_predictions).T

predictions = np.array(predictions).T
predictions_proba = np.array(predictions_proba).T
best_features_predictions = np.array(best_features_predictions).T
best_features_predictions_proba = np.array(best_features_predictions_proba).T
print(best_features_predictions_proba)

# Transponer las listas colocando los datos en columnas para cada patrón
train_set_labels_np_matrix = np.c_[tuple(train_set_labels[pattern] for pattern in patterns_list)]
test_set_labels_np_matrix = np.c_[tuple(test_set_labels[pattern] for pattern in patterns_list)]

# # Mostrar las medidas de rendimiento
print("\nRendimiento del modelo entrenado")
mlutils.model_performance_data(train_set_labels_np_matrix, model_predictions, patterns_list)

# # Mostrar las medidas de rendimiento
print(f"\nMétricas seleccionadas: {best_features}")
print(f"[{len(best_features['p.initialization'])}, {len(best_features['p.superposition'])}, {len(best_features['p.oracle'])}]")
print("Rendimiento del modelo mejorado")
mlutils.model_performance_data(train_set_labels_np_matrix, model_best_features_predictions, patterns_list)

# # Mostrar las medidas de rendimiento
print("\nRendimiento del modelo con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)

print(f"\nMétricas seleccionadas: {best_features}")
print(f"[{len(best_features['p.initialization'])}, {len(best_features['p.superposition'])}, {len(best_features['p.oracle'])}]")
print("\nRendimiento del modelo mejorado con el conjunto de prueba")
mlutils.model_performance_data(test_set_labels_np_matrix, best_features_predictions, patterns_list)

# Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
for i in range(0,test_results_num):
    print(f"\nInstancia {i+1}: {test_set_labels.iloc[i].tolist()}")
    for j in range(len(predictions_proba[0][i])):
        print(f"Patrón {patterns_list[j]}: ")
        if predictions_proba[1][i][j] > predictions_proba[0][i][j]:
            print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[1][i][j]*100, 2)}%")
        else:
            print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[0][i][j]*100, 2)}%")

        if best_features_predictions_proba[1][i][j] > best_features_predictions_proba[0][i][j]:
            print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[1][i][j]*100, 2)}%")
        else:
            print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[0][i][j]*100, 2)}%")