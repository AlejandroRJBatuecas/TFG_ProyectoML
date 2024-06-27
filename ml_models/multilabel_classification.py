import pandas as pd
import numpy as np
import joblib
import os

from .ml_utils import show_data_structure, get_correlation_matrix, model_performance_data
from pathlib import Path
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
min_importance_value = 0.01 # Selecciona características con una importancia superior a este valor
min_correlation_value = 0.5 # Selecciona características con una correlación superior a este valor
cv_value = 3 # Por defecto = 5. Número de particiones realizadas en la validación cruzada. Ponemos 3 ya que es un conjunto de datos pequeño
test_results_num = 10 # Número de registros de prueba mostrados
trained_model_path = "./trained_models/kn_classifier.joblib" # Ruta de almacenamiento del modelo (respecto al directorio raíz)
best_features_trained_model_path = "./trained_models/best_features_kn_classifier.joblib" # Ruta de almacenamiento del modelo con mejores características (respecto al directorio raíz)

# Establecer la semilla de aleatoridad
np.random.seed(42)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def prepare_data(data):
    # Limpiar las filas con algún dato nulo
    data = data.dropna()

    # Elimnar las columnas relacionadas con Oracle
    data = data.drop(columns=eliminated_metrics)

    # Separar datos y etiquetas
    data_values = data.select_dtypes(include=[np.number])
    data_labels = data[patterns_list]

    return data_values, data_labels

def get_datasets(data, data_values, data_labels):
    # Obtener el conjunto de entrenamiento y de prueba
    train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(data_values, data_labels, test_size=test_set_size, stratify=data_labels)

    # Mostrar la estructura de los datos
    show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
    #create_data_histogram(data)

    # Ver la correlación entre los datos
    data_vars = data.drop(["id", "language", "extension", "author", "name", "path", "circuit"], axis=1)
    # # Obtener la matriz de correlación
    get_correlation_matrix(data_vars, min_correlation_value, patterns_list)

    return train_set_values, test_set_values, train_set_labels, test_set_labels

def create_ml_model(train_set_values, train_set_labels):
    # Definir la cuadrícula de hiperparámetros a buscar
    param_grid = {
        'classifier__n_neighbors': [1, 3, 5, 7, 9], # Mejor valores impares para evitar empates
        'classifier__weights': ['uniform', 'distance']
    }

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
    best_features_pipeline.steps[1][1].threshold = min_importance_value

    # Transponer las listas colocando los datos en columnas para cada patrón
    train_set_labels_np_matrix = np.c_[tuple(train_set_labels[pattern] for pattern in patterns_list)]

    # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_value)
    grid_search.fit(train_set_values, train_set_labels_np_matrix)
    print("\nMejores hiperparámetros para Normal: \n", grid_search.best_params_)

    best_features_grid_search = GridSearchCV(best_features_pipeline, param_grid, cv=cv_value)
    best_features_grid_search.fit(train_set_values, train_set_labels_np_matrix)
    print("\nMejores hiperparámetros para Mejorado: \n", best_features_grid_search.best_params_)

    # Obtener el mejor modelo
    knn_classifier = grid_search.best_estimator_
    best_features_knn_classifier = best_features_grid_search.best_estimator_

    return pipeline, best_features_pipeline, knn_classifier, best_features_knn_classifier, train_set_labels_np_matrix

# Obtener la importancia de las caracteristicas
def get_feature_importance(data_values, best_features_knn_classifier):
    # Obtener los nombres de las características originales
    feature_names = data_values.columns
    # Obtener el selector de características desde la pipeline
    feature_selector = best_features_knn_classifier.named_steps['feature_selection']
    # Obtener el modelo subyacente desde el selector de características
    model = feature_selector.estimator_
    # Obtener las importancias de las características
    importances = model.feature_importances_
    # Crear un diccionario de características e importancias
    feature_importance_dict = dict(zip(feature_names, importances))
    # Ordenar las características por importancia
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    # Imprimir las importancias de las características
    print("\nImportancias de las características:")
    for feature, importance in sorted_features:
        print(f"{feature}: {round(importance*100, 3)}%")

    # Filtrar las características originales para obtener solo las seleccionadas
    selected_features = best_features_knn_classifier.named_steps['feature_selection'].get_support()
    best_features = [feature for feature, selected in zip(feature_names, selected_features) if selected]

    return best_features

# Evaluar el rendimiento del modelo
def evaluate_model_performance(knn_classifier, train_set_values, train_set_labels_np_matrix):
    # Calcular las predicciones del clasificador mediante evaluación cruzada
    predictions = cross_val_predict(knn_classifier, train_set_values, train_set_labels_np_matrix, cv=cv_value)
    # Mostrar las medidas de rendimiento
    print("\nRendimiento del modelo entrenado")
    model_performance_data(train_set_labels_np_matrix, predictions, patterns_list)

# Evaluar el rendimiento del modelo con las mejores métricas
def evaluate_best_features_model_performance(best_features_knn_classifier, train_set_values, train_set_labels_np_matrix, best_features):
    # Calcular las predicciones del clasificador mediante evaluación cruzada
    predictions = cross_val_predict(best_features_knn_classifier, train_set_values, train_set_labels_np_matrix, cv=cv_value)
    # Mostrar las medidas de rendimiento
    print(f"\nMétricas seleccionadas: {len(best_features)}\n{best_features}")
    print("\nRendimiento del modelo mejorado")
    model_performance_data(train_set_labels_np_matrix, predictions, patterns_list)

# Evaluar el rendimiento del modelo con el conjunto de prueba
def test_model_performance(knn_classifier, test_set_values, test_set_labels):
    # Realizar predicciones en el conjunto de prueba
    predictions = knn_classifier.predict(test_set_values)
    predictions_proba = knn_classifier.predict_proba(test_set_values)
    # Transponer las listas colocando los datos en columnas para cada patrón
    test_set_labels_np_matrix = np.c_[tuple(test_set_labels[pattern] for pattern in patterns_list)]
    # Mostrar las medidas de rendimiento
    print("\nRendimiento del modelo con el conjunto de prueba")
    model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)
    # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
    for i in range(len(test_set_labels)):
        test_set_labels_list = test_set_labels.iloc[i].tolist()
        for j in range(len(test_set_labels_list)):
            if (predictions_proba[j][i][1] > predictions_proba[j][i][0]):
                if test_set_labels_list[j] == False:
                    print(f"\nModelo normal --> Cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][1]*100, 3)}%")
                    print(f"Instancia {i+1}: {test_set_labels_list[j]}")
            else:
                if test_set_labels_list[j] == True:
                    print(f"\nModelo normal --> No cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][0]*100, 3)}%")
                    print(f"Instancia {i+1}: {test_set_labels_list[j]}")

    return predictions_proba, test_set_labels_np_matrix

# Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
def test_best_features_model_performance(best_features_knn_classifier, test_set_values, best_features, test_set_labels_np_matrix, test_set_labels):
    # Realizar predicciones en el conjunto de prueba
    predictions = best_features_knn_classifier.predict(test_set_values)
    best_features_predictions_proba = best_features_knn_classifier.predict_proba(test_set_values)
    # # Mostrar las medidas de rendimiento
    print(f"\nMétricas seleccionadas: {len(best_features)}\n{best_features}")
    print("\nRendimiento del modelo mejorado con el conjunto de prueba")
    model_performance_data(test_set_labels_np_matrix, predictions, patterns_list)
    # # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
    for i in range(len(test_set_labels)):
        test_set_labels_list = test_set_labels.iloc[i].tolist()
        for j in range(len(test_set_labels_list)):
            if best_features_predictions_proba[j][i][1] > best_features_predictions_proba[j][i][0]:
                if test_set_labels_list[j] == False:
                    print(f"\nModelo mejorado --> Cumple el patrón {patterns_list[j]} en un {round(best_features_predictions_proba[j][i][1]*100, 3)}%")
                    print(f"Instancia {i+1}: {test_set_labels_list[j]}")
            else:
                if test_set_labels_list[j] == True:
                    print(f"\nModelo mejorado --> No cumple el patrón {patterns_list[j]} en un {round(best_features_predictions_proba[j][i][0]*100, 3)}%")
                    print(f"Instancia {i+1}: {test_set_labels_list[j]}")

    return best_features_predictions_proba

# Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
def show_first_test_predictions(test_set_labels, predictions_proba, best_features_predictions_proba):
    print("\nResultados de los primeros", test_results_num, "registros de prueba:")
    for i in range(0, test_results_num):
        print(f"\nInstancia {i+1}: {test_set_labels.iloc[i].tolist()}")
        for j in range(len(predictions_proba)):
            print(f"Patrón {patterns_list[j]}: ")
            if predictions_proba[j][i][1] > predictions_proba[j][i][0]:
                print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[j][i][1]*100, 3)}%")
            else:
                print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[j][i][0]*100, 3)}%")

            if best_features_predictions_proba[j][i][1] > best_features_predictions_proba[j][i][0]:
                print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[j][i][1]*100, 3)}%")
            else:
                print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[j][i][0]*100, 3)}%")

# Realizar predicción a partir de datos externos
def generate_prediction(test_data, knn_classifier, best_features_knn_classifier):
    # Obtener predicciones y probabilidades de los datos de prueba con el modelo entrenado
    test_data_pred = knn_classifier.predict(test_data)
    predictions_proba = knn_classifier.predict_proba(test_data)
    # Realizar predicciones con los datos de prueba
    best_features_test_data_pred = best_features_knn_classifier.predict(test_data)
    best_features_predictions_proba = best_features_knn_classifier.predict_proba(test_data)
    # Imprimir los porcentajes de predicción
    for i in range(len(predictions_proba[0])):
        print(f"\nInstancia {i+1}:")
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

def store_model(knn_classifier, best_features_knn_classifier):
    # Separar el directorio y el nombre del archivo
    file_folder, _ = os.path.split(trained_model_path)
    best_features_file_folder, _ = os.path.split(best_features_trained_model_path)

    print(file_folder)

    # Crear el directorio si no existe
    os.makedirs(file_folder, exist_ok=True)
    os.makedirs(best_features_file_folder, exist_ok=True)

    # Almacenar los modelos entrenados en un archivo
    joblib.dump(knn_classifier, trained_model_path)
    joblib.dump(best_features_knn_classifier, best_features_trained_model_path)

def generate_ml_models(data_filename):
    # Leer el csv con los datos de entrenamiento
    data = pd.read_csv(Path(data_filename), delimiter=";")

    data_values, data_labels = prepare_data(data)

    train_set_values, test_set_values, train_set_labels, test_set_labels = get_datasets(data, data_values, data_labels)

    pipeline, best_features_pipeline, knn_classifier, best_features_knn_classifier, train_set_labels_np_matrix = create_ml_model(train_set_values, train_set_labels)

    # Almacenar los modelos entrenados en un archivo
    store_model(knn_classifier, best_features_knn_classifier)

    best_features = get_feature_importance(data_values, best_features_knn_classifier)

    return train_set_values, test_set_values, train_set_labels, test_set_labels, pipeline, best_features_pipeline, knn_classifier, best_features_knn_classifier, train_set_labels_np_matrix, best_features

def show_model_evaluation(data_filename):
    train_set_values, test_set_values, _, test_set_labels, _, _, knn_classifier, best_features_knn_classifier, train_set_labels_np_matrix, best_features = generate_ml_models(data_filename)
    
    evaluate_model_performance(knn_classifier, train_set_values, train_set_labels_np_matrix)
    evaluate_best_features_model_performance(best_features_knn_classifier, train_set_values, train_set_labels_np_matrix, best_features)

    predictions_proba, test_set_labels_np_matrix = test_model_performance(knn_classifier, test_set_values, test_set_labels)
    best_features_predictions_proba = test_best_features_model_performance(best_features_knn_classifier, test_set_values, best_features, test_set_labels_np_matrix, test_set_labels)

    show_first_test_predictions(test_set_labels, predictions_proba, best_features_predictions_proba)

def get_prediction(data_filename, test_data_filename):
    # Si existen los modelos entrenados, los recuperamos. Si no, los generamos
    if Path(trained_model_path).is_file() and Path(best_features_trained_model_path).is_file():
        knn_classifier = joblib.load(trained_model_path)
        best_features_knn_classifier = joblib.load(best_features_trained_model_path)
    else:
        _, _, _, _, _, _, knn_classifier, best_features_knn_classifier, _, _ = generate_ml_models(data_filename)
    
    # Obtener los datos a predecir
    test_data = pd.read_csv(Path(test_data_filename), delimiter=";")

    # Generar las predicciones
    generate_prediction(test_data, knn_classifier, best_features_knn_classifier)

if __name__ == "__main__":
    data_filename = "../datasets/dataset_openqasm_qiskit.csv"
    show_model_evaluation(data_filename)