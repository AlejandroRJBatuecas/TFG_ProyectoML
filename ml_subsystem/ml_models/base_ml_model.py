import pandas as pd
import numpy as np

from config import ml_parameters
from .ml_utils import show_data_structure, get_correlation_matrix
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.model_selection import train_test_split

class BaseMLModel(ABC):
    def __init__(self, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Definir el modelo de ML utilizado y los parámetros para la búsqueda de hiperparámetros
        self.model, self.param_grid = self._init_model()
        # Leer el csv con los datos de entrenamiento
        self.data = pd.read_csv(Path(data_filename), delimiter=";")
        self.test_size = test_size
        self.data_values, self.data_labels = self._prepare_data()
        self.train_set_values, self.test_set_values, self.train_set_labels, self.test_set_labels = self._get_datasets()

    # Definir el modelo de ML utilizado y los parámetros para la búsqueda de hiperparámetros
    @abstractmethod
    def _init_model(self):
        pass

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
        print(self.data_values)
        print(self.data_labels)

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
    @abstractmethod
    def _create_pipelines(self):
        pass

    # Obtener los clasificadores con los mejores hiperparámetros
    @abstractmethod
    def _get_classifiers(self):
        pass
    
    # Obtener la importancia de las caracteristicas
    @abstractmethod
    def _get_feature_importance(self):
        pass
    
    # Evaluar el rendimiento del modelo
    @abstractmethod
    def _evaluate_model_performance(self):
        pass
    
    # Evaluar el rendimiento del modelo con las mejores métricas
    @abstractmethod
    def _evaluate_best_features_model_performance(self):
        pass
    
    # Evaluar el rendimiento del modelo con el conjunto de prueba
    @abstractmethod
    def _test_model_performance(self):
        pass
    
    # Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
    @abstractmethod
    def _test_best_features_model_performance(self):
        pass
    
    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
    @abstractmethod
    def _show_first_test_predictions(self, test_results_num=ml_parameters.test_results_num):
        pass
    
    # Mostrar los valores de rendimiento de todos los modelos
    def show_model_evaluation(self, test_results_num=ml_parameters.test_results_num):
        # Evaluar el rendimiento del modelo
        self._evaluate_model_performance()
        self._evaluate_best_features_model_performance()
        # Evaluar el modelo con el conjunto de prueba
        self._test_model_performance()
        self._test_best_features_model_performance()
        # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
        self._show_first_test_predictions(test_results_num)
    
    # Realizar predicción a partir de datos externos
    @abstractmethod
    def get_prediction(self):
        pass