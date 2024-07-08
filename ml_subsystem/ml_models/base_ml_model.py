import pandas as pd
import numpy as np

from config import ml_parameters
from .ml_utils import show_data_structure, get_correlation_matrix
from pathlib import Path
from sklearn.model_selection import train_test_split

class BaseMLModel:
    def __init__(self, param_grid, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Inicialización de parámetros
        self.param_grid = param_grid
        # Leer el csv con los datos de entrenamiento
        self.data = pd.read_csv(Path(data_filename), delimiter=";")
        self.test_size = test_size

        self.data_values, self.data_labels = self._prepare_data()
        self.train_set_values, self.test_set_values, self.train_set_labels, self.test_set_labels = self._get_datasets()
        
        # Transponer las listas colocando los datos en columnas para cada patrón
        self.train_set_labels_np_matrix = np.c_[tuple(self.train_set_labels[pattern] for pattern in ml_parameters.patterns_list)]
        # Transponer las listas colocando los datos en columnas para cada patrón
        self.test_set_labels_np_matrix = np.c_[tuple(self.test_set_labels[pattern] for pattern in ml_parameters.patterns_list)]
        self.pipeline, self.best_features_pipeline, self.classifier, self.best_features_classifier = self._create_ml_model()
        self.best_features = self._get_feature_importance()
        self._evaluate_model_performance()
        self._evaluate_best_features_model_performance()
        self.predictions_proba = self._test_model_performance()
        self.best_features_predictions_proba = self._test_best_features_model_performance()
        self._show_first_test_predictions()

    # Realizar el preprocesamiento de los datos
    def _prepare_data(self):
        # Limpiar las filas con algún dato nulo
        self.data = self.data.dropna()

        # Elimnar las columnas relacionadas con Oracle
        self.data = self.data.drop(columns=ml_parameters.eliminated_metrics)

        # Separar datos y etiquetas
        data_values = self.data.select_dtypes(include=[np.number])
        data_labels = self.data[ml_parameters.patterns_list]

        return data_values, data_labels
    
    # Obtener los valores y etiquetas en conjuntos de entrenamiento y prueba
    def _get_datasets(self):
        # Obtener el conjunto de entrenamiento y de prueba
        train_set_values, test_set_values, train_set_labels, test_set_labels = train_test_split(self.data_values, self.data_labels, test_size=ml_parameters.test_set_size, random_state=ml_parameters.random_state_value, stratify=self.data_labels)

        # Mostrar la estructura de los datos
        show_data_structure(self.data, self.data_values, self.data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels)
        #create_data_histogram(data)

        # Ver la correlación entre los datos
        data_vars = self.data.drop(ml_parameters.eliminated_columns, axis=1)
        # # Obtener la matriz de correlación
        get_correlation_matrix(data_vars, ml_parameters.min_correlation_value, ml_parameters.patterns_list)

        return train_set_values, test_set_values, train_set_labels, test_set_labels
    
    # Definir las pipelines
    def _create_pipelines(self):
        raise NotImplementedError

    # Obtener los clasificadores con los mejores hiperparámetros
    def _get_classifiers(self):
        raise NotImplementedError
    
    # Obtener la importancia de las caracteristicas
    def _get_feature_importance(self):
        raise NotImplementedError
    
    # Evaluar el rendimiento del modelo
    def _evaluate_model_performance(self):
        raise NotImplementedError
    
    # Evaluar el rendimiento del modelo con las mejores métricas
    def _evaluate_best_features_model_performance(self):
        raise NotImplementedError
    
    # Evaluar el rendimiento del modelo con el conjunto de prueba
    def _test_model_performance(self):
        raise NotImplementedError
    
    # Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
    def _test_best_features_model_performance(self):
        raise NotImplementedError
    
    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
    def _show_first_test_predictions(self, test_results_num=ml_parameters.test_results_num):
        raise NotImplementedError
    
    # Mostrar los valores de rendimiento de todos los modelos
    def show_model_evaluation(self, test_results_num=ml_parameters.test_results_num):
        raise NotImplementedError
    
    # Realizar predicción a partir de datos externos
    def get_prediction(self):
        raise NotImplementedError