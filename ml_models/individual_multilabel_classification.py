import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.calibration import cross_val_predict

# Parámetros generales
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
patterns_list = ["p.initialization", "p.superposition", "p.oracle"] # Lista de patrones a detectar
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"] # Métricas de Oráculo eliminadas
min_importance_value = [0.02, 0.02, 0.01] # Selecciona características con una importancia superior a este valor
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
cv_value = 3 # Por defecto = 5. Número de particiones realizadas en la validación cruzada. Ponemos 3 ya que es un conjunto de datos pequeño
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

# Obtener el conjunto de entrenamiento y de prueba
train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(data_values, data_labels, test_size=test_set_size, stratify=data_labels)

# Mostrar la estructura de los datos
mlutils.show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
#mlutils.create_data_histogram(data)

# Ver la correlación entre los datos
data_vars = data.drop(["id", "language", "extension", "author", "name", "path", "circuit"], axis=1)
# # Obtener la matriz de correlación
mlutils.get_correlation_matrix(data_vars, min_correlation_value, patterns_list)

# Definir la cuadrícula de hiperparámetros a buscar
param_grid = {
    'classifier__n_neighbors': [1, 3, 5, 7, 9], # Mejor valores impares para evitar empates
    'classifier__weights': ['uniform', 'distance']
}

# Para cada patrón, crear y entrenar un clasificador
for i, pattern in enumerate(patterns_list):
    # Obtener las etiquetas del patrón
    pattern_train_labels = train_set_labels[pattern]
    pattern_test_labels = test_set_labels[pattern]

    # Crear el pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ])

    print("\nHiperparámetros por defecto de KNeighborsClassifier:\n", 
        pipeline.named_steps['classifier'].get_params())

    best_features_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
        ('classifier', KNeighborsClassifier())
    ])

    # Asignar el umbral de importancia al selector de características
    best_features_pipeline.steps[1][1].threshold = min_importance_value[i]

    # Transponer las listas colocando los datos en columnas para cada patrón
    #train_set_labels_np_matrix = np.c_[tuple(train_set_labels[pattern] for pattern in patterns_list)]

    # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_value)
    grid_search.fit(train_set_values, pattern_train_labels)
    print("\nMejores hiperparámetros para Normal: \n", grid_search.best_params_)

    best_features_grid_search = GridSearchCV(best_features_pipeline, param_grid, cv=cv_value)
    best_features_grid_search.fit(train_set_values, pattern_train_labels)
    print("\nMejores hiperparámetros para Mejorado: \n", best_features_grid_search.best_params_)

    # Obtener el mejor modelo
    knn_classifier = grid_search.best_estimator_
    best_features_knn_classifier = best_features_grid_search.best_estimator_

    # Filtrar las características originales para obtener solo las seleccionadas
    selected_features = best_features_knn_classifier.named_steps['feature_selection'].get_support()
    feature_names = data_values.columns
    best_features = [feature for feature, selected in zip(feature_names, selected_features) if selected]

    # Evaluar el rendimiento del modelo
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    predictions = cross_val_predict(knn_classifier, train_set_values, pattern_train_labels, cv=cv_value)
    # # Mostrar las medidas de rendimiento
    print(f"\nRendimiento del clasificador de {pattern}")
    mlutils.model_performance_data(pattern_train_labels, predictions, pattern)

    # Evaluar el rendimiento del modelo con las mejores métricas
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    predictions = cross_val_predict(best_features_knn_classifier, train_set_values, pattern_train_labels, cv=cv_value)
    # # Mostrar las medidas de rendimiento
    print(f"\nMétricas seleccionadas: {len(best_features)}\n{best_features}")
    print("\nRendimiento del modelo mejorado")
    mlutils.model_performance_data(pattern_train_labels, predictions, pattern)

    # Transponer las listas colocando los datos en columnas para cada patrón
    #test_set_labels_np_matrix = np.c_[tuple(test_set_labels[pattern] for pattern in patterns_list)]

    # Evaluar el rendimiento del modelo con el conjunto de prueba
    # # Realizar predicciones en el conjunto de prueba
    predictions = knn_classifier.predict(test_set_values)
    predictions_proba = knn_classifier.predict_proba(test_set_values)
    # # Mostrar las medidas de rendimiento
    print("\nRendimiento del modelo con el conjunto de prueba")
    mlutils.model_performance_data(pattern_test_labels, predictions, pattern)
    # # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
    pattern_test_labels_list = pattern_test_labels.tolist()
    for j in range(len(pattern_test_labels_list)):
        if predictions_proba[j][1] > predictions_proba[j][0]:
            if pattern_test_labels_list[j] == False:
                print(f"\nModelo normal --> Cumple el patrón en un {round(predictions_proba[j][1]*100, 3)}%")
                print(f"Instancia {j+1}: {pattern_test_labels_list[j]}")
        else:
            if pattern_test_labels_list[j] == True:
                print(f"\nModelo normal --> No cumple el patrón en un {round(predictions_proba[j][0]*100, 3)}%")
                print(f"Instancia {j+1}: {pattern_test_labels_list[j]}")

    # Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
    # # Realizar las predicciones en el conjunto de prueba con las mejores métricas
    predictions = best_features_knn_classifier.predict(test_set_values)
    best_features_predictions_proba = best_features_knn_classifier.predict_proba(test_set_values)
    # # Mostrar las medidas de rendimiento
    print(f"\nMétricas seleccionadas: {len(best_features)}\n{best_features}")
    print("\nRendimiento del modelo mejorado con el conjunto de prueba")
    mlutils.model_performance_data(pattern_test_labels, predictions, pattern)
    # # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
    for j in range(len(pattern_test_labels_list)):
        if best_features_predictions_proba[j][1] > best_features_predictions_proba[j][0]:
            if pattern_test_labels_list[j] == False:
                print(f"\nModelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[j][1]*100, 3)}%")
                print(f"Instancia {j+1}: {pattern_test_labels_list[j]}")
        else:
            if pattern_test_labels_list[j] == True:
                print(f"\nModelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[j][0]*100, 3)}%")
                print(f"Instancia {j+1}: {pattern_test_labels_list[j]}")

    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
    print("\nResultados de los primeros", test_results_num, "registros de prueba:")
    for j in range(0, test_results_num):
        print(f"\nInstancia {j+1}: {pattern_test_labels_list[j]}")
        if predictions_proba[j][1] > predictions_proba[j][0]:
            print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[j][1]*100, 3)}%")
        else:
            print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[j][0]*100, 3)}%")

        if best_features_predictions_proba[j][1] > best_features_predictions_proba[j][0]:
            print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[j][1]*100, 3)}%")
        else:
            print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[j][0]*100, 3)}%")