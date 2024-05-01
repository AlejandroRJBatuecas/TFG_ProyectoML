import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import cross_val_predict

# Parámetros generales
test_set_size = 0.3 # Porcentaje de datos utilizados para el conjunto de prueba
patterns_list = ["p.initialization", "p.superposition", "p.oracle"] # Lista de patrones a detectar
eliminated_metrics = ["m.NoOr", "m.NoCOr", "m.%QInOr", "m.%QInCOr", "m.AvgOrD", "m.MaxOrD"] # Métricas de Oráculo eliminadas
min_importance_values = [0.02, 0.03, 0.03] # Selecciona características con una importancia superior a este valor
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
print(data_values)
# Obtener el conjunto de entrenamiento y de prueba
train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(data_values, data_labels, test_size=test_set_size, stratify=data_labels)

# Mostrar la estructura de los datos
mlutils.show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
#mlutils.create_data_histogram(data)

# Ver la correlación entre los datos
data_vars = data.drop(["id", "language", "extension", "author", "name", "path", "circuit"], axis=1)
# # Obtener la matriz de correlación
mlutils.get_correlation_matrix(data_vars, min_correlation_value, patterns_list)

# Parámetros para la búsqueda de hiperparámetros
params = [
    {'classifier__n_estimators': [50, 100, 200], 'classifier__min_samples_leaf': [10]},
    {'classifier__n_estimators': [50, 100, 200], 'classifier__min_samples_leaf': [20]},
    {'classifier__n_estimators': [50, 100, 200], 'classifier__min_samples_leaf': [2]}
]
    
# Para cada patrón, crear y entrenar un clasificador
for i, pattern in enumerate(patterns_list):
    # Obtener las etiquetas del patrón
    pattern_train_labels = train_set_labels[pattern]
    pattern_test_labels = test_set_labels[pattern]

    # Crear el pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    print(pipeline.named_steps['classifier'].get_params())

    best_features_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(RandomForestRegressor())),
        ('classifier', RandomForestClassifier())
    ])

    # Asignar el umbral de importancia al selector de características
    best_features_pipeline.steps[1][1].threshold = min_importance_values[i]
    
    # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
    print("Mejores hiperparámetros para los clasificadores de: ", pattern)
    grid_search = GridSearchCV(pipeline, params[i], cv=5)
    grid_search.fit(train_set_values, pattern_train_labels)
    print("Base: ", grid_search.best_params_)

    best_features_grid_search = GridSearchCV(best_features_pipeline, params[i], cv=5)
    best_features_grid_search.fit(train_set_values, pattern_train_labels)
    print("Mejorado: ", best_features_grid_search.best_params_)

    classifier = grid_search.best_estimator_
    best_features_classifier = best_features_grid_search.best_estimator_

    # Obtener las características seleccionadas
    selected_features = best_features_classifier.named_steps['feature_selection'].get_support()

    # Filtrar las características originales para obtener solo las seleccionadas
    features_names = data_values.columns
    best_features = [feature for feature, selected in zip(features_names, selected_features) if selected]
    
    # Realizar las predicciones con el clasificador
    predictions = classifier.predict(test_set_values)
    predictions_proba = classifier.predict_proba(test_set_values)
    best_features_predictions = best_features_classifier.predict(test_set_values)
    best_features_predictions_proba = best_features_classifier.predict_proba(test_set_values)

    # Evaluar el rendimiento del modelo
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    cross_val_predictions = cross_val_predict(classifier, train_set_values, pattern_train_labels, cv=cv_value)
    # # Mostrar las medidas de rendimiento
    print(f"\nRendimiento del clasificador de {pattern}")
    mlutils.model_performance_data(pattern_train_labels, cross_val_predictions, patterns_list)

    # Evaluar el rendimiento del modelo con las mejores métricas
    # # Calcular las predicciones del clasificador mediante evaluación cruzada
    best_features_cross_val_predictions = cross_val_predict(best_features_classifier, train_set_values, pattern_train_labels, cv=cv_value)
    # # Mostrar las medidas de rendimiento
    print(f"\nMétricas seleccionadas ({len(best_features)}): {best_features}")
    print("Rendimiento del modelo mejorado")
    mlutils.model_performance_data(pattern_train_labels, best_features_cross_val_predictions, patterns_list)

    # # Mostrar las medidas de rendimiento
    print("\nRendimiento del modelo con el conjunto de prueba")
    mlutils.model_performance_data(pattern_test_labels, predictions, patterns_list)

    print(f"\nMétricas seleccionadas ({len(best_features)}): {best_features}")
    print("\nRendimiento del modelo mejorado con el conjunto de prueba")
    mlutils.model_performance_data(pattern_test_labels, best_features_predictions, patterns_list)

    pattern_test_labels_list = pattern_test_labels.tolist()

    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
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