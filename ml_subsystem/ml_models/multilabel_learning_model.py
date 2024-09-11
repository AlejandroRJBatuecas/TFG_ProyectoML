import pandas as pd
import numpy as np

from config import ml_parameters
from .base_ml_model import BaseMLModel
from .ml_utils import model_performance_data
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import cross_val_predict

class MultilabelLearningModel(BaseMLModel):
    def __init__(self, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Inicialización de parámetros heredados
        super().__init__(data_filename, test_size)
        # Transponer las listas colocando los datos en columnas para cada patrón
        self.train_set_labels_np_matrix = np.c_[tuple(self.train_set_labels[pattern] for pattern in ml_parameters.patterns_list)]
        # Transponer las listas colocando los datos en columnas para cada patrón
        self.test_set_labels_np_matrix = np.c_[tuple(self.test_set_labels[pattern] for pattern in ml_parameters.patterns_list)]
        self.pipeline, self.best_features_pipeline = self._create_pipelines()
        self.classifier, self.best_features_classifier = self._get_classifiers()
        self.best_features = self._get_feature_importance()
        self.predictions_proba = self.classifier.predict_proba(self.test_set_values)
        self.best_features_predictions_proba = self.best_features_classifier.predict_proba(self.test_set_values)

    # Definir las pipelines
    def _create_pipelines(self):
        # Crear el pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.model)
        ], memory=None)

        print(f"\nHiperparámetros por defecto de {self.model.__class__.__name__}:\n", 
            pipeline.named_steps['classifier'].get_params())

        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=ml_parameters.random_state_value
        ) # Inicializado con los valores por defecto

        best_features_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(rf_clf)),
            ('classifier', self.model)
        ], memory=None)

        # Asignar el umbral de importancia al selector de características
        best_features_pipeline.steps[1][1].threshold = ml_parameters.min_importance_value

        return pipeline, best_features_pipeline

    # Obtener los clasificadores con los mejores hiperparámetros
    def _get_classifiers(self):
        # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=ml_parameters.cv_value)
        grid_search.fit(self.train_set_values, self.train_set_labels_np_matrix)
        print("\nMejores hiperparámetros para Normal: \n", grid_search.best_params_)

        best_features_grid_search = GridSearchCV(self.best_features_pipeline, self.param_grid, cv=ml_parameters.cv_value)
        best_features_grid_search.fit(self.train_set_values, self.train_set_labels_np_matrix)
        print("\nMejores hiperparámetros para Mejorado: \n", best_features_grid_search.best_params_)

        # Obtener el mejor modelo
        classifier = grid_search.best_estimator_
        best_features_classifier = best_features_grid_search.best_estimator_

        return classifier, best_features_classifier

    # Obtener la importancia de las caracteristicas
    def _get_feature_importance(self):
        # Obtener los nombres de las características originales
        feature_names = self.data_values.columns
        # Obtener el selector de características desde la pipeline
        feature_selector = self.best_features_classifier.named_steps['feature_selection']
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
        selected_features = self.best_features_classifier.named_steps['feature_selection'].get_support()
        best_features = [feature for feature, selected in zip(feature_names, selected_features) if selected]

        return best_features
    
    # Evaluar el rendimiento del modelo
    def _evaluate_model_performance(self):
        # Calcular las predicciones del clasificador mediante evaluación cruzada
        predictions = cross_val_predict(self.classifier, self.train_set_values, self.train_set_labels_np_matrix, cv=ml_parameters.cv_value)
        # Mostrar las medidas de rendimiento
        print("\nRendimiento del modelo entrenado")
        model_performance_data(self.train_set_labels_np_matrix, predictions, ml_parameters.patterns_list)

    # Evaluar el rendimiento del modelo con las mejores métricas
    def _evaluate_best_features_model_performance(self):
        # Calcular las predicciones del clasificador mediante evaluación cruzada
        predictions = cross_val_predict(self.best_features_classifier, self.train_set_values, self.train_set_labels_np_matrix, cv=ml_parameters.cv_value)
        # Mostrar las medidas de rendimiento
        print(f"\nMétricas seleccionadas: {len(self.best_features)}\n{self.best_features}")
        print("\nRendimiento del modelo mejorado")
        model_performance_data(self.train_set_labels_np_matrix, predictions, ml_parameters.patterns_list)
    
    # Evaluar el rendimiento del modelo con el conjunto de prueba
    def _test_model_performance(self):
        # Realizar predicciones en el conjunto de prueba
        predictions = self.classifier.predict(self.test_set_values)
        
        # Mostrar las medidas de rendimiento
        print("\nRendimiento del modelo con el conjunto de prueba")
        model_performance_data(self.test_set_labels_np_matrix, predictions, ml_parameters.patterns_list)
        # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
        for i in range(len(self.test_set_labels)):
            test_set_labels_list = self.test_set_labels.iloc[i].tolist()
            for j in range(len(test_set_labels_list)):
                if (self.predictions_proba[j][i][1] > self.predictions_proba[j][i][0]):
                    if test_set_labels_list[j] == False:
                        print(f"\nModelo normal --> Cumple el patrón {ml_parameters.patterns_list[j]} en un {round(self.predictions_proba[j][i][1]*100, 3)}%")
                        print(f"Instancia {i+1}: {test_set_labels_list[j]}")
                else:
                    if test_set_labels_list[j] == True:
                        print(f"\nModelo normal --> No cumple el patrón {ml_parameters.patterns_list[j]} en un {round(self.predictions_proba[j][i][0]*100, 3)}%")
                        print(f"Instancia {i+1}: {test_set_labels_list[j]}")
    
    # Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
    def _test_best_features_model_performance(self):
        # Realizar predicciones en el conjunto de prueba
        predictions = self.best_features_classifier.predict(self.test_set_values)
        # # Mostrar las medidas de rendimiento
        print(f"\nMétricas seleccionadas: {len(self.best_features)}\n{self.best_features}")
        print("\nRendimiento del modelo mejorado con el conjunto de prueba")
        model_performance_data(self.test_set_labels_np_matrix, predictions, ml_parameters.patterns_list)
        # # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
        for i in range(len(self.test_set_labels)):
            test_set_labels_list = self.test_set_labels.iloc[i].tolist()
            for j in range(len(test_set_labels_list)):
                if self.best_features_predictions_proba[j][i][1] > self.best_features_predictions_proba[j][i][0]:
                    if test_set_labels_list[j] == False:
                        print(f"\nModelo mejorado --> Cumple el patrón {ml_parameters.patterns_list[j]} en un {round(self.best_features_predictions_proba[j][i][1]*100, 3)}%")
                        print(f"Instancia {i+1}: {test_set_labels_list[j]}")
                else:
                    if test_set_labels_list[j] == True:
                        print(f"\nModelo mejorado --> No cumple el patrón {ml_parameters.patterns_list[j]} en un {round(self.best_features_predictions_proba[j][i][0]*100, 3)}%")
                        print(f"Instancia {i+1}: {test_set_labels_list[j]}")
    
    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
    def _show_first_test_predictions(self, test_results_num=ml_parameters.test_results_num):
        print("\nResultados de los primeros", test_results_num, "registros de prueba:")
        for i in range(0, test_results_num):
            print(f"\nInstancia {i+1}: {self.test_set_labels.iloc[i].tolist()}")
            for j in range(len(self.predictions_proba)):
                print(f"Patrón {ml_parameters.patterns_list[j]}: ")
                if self.predictions_proba[j][i][1] > self.predictions_proba[j][i][0]:
                    print(f"Modelo normal --> Cumple el patrón en un {round(self.predictions_proba[j][i][1]*100, 3)}%")
                else:
                    print(f"Modelo normal --> No cumple el patrón en un {round(self.predictions_proba[j][i][0]*100, 3)}%")

                if self.best_features_predictions_proba[j][i][1] > self.best_features_predictions_proba[j][i][0]:
                    print(f"Modelo mejorado --> Cumple el patrón en un {round(self.best_features_predictions_proba[j][i][1]*100, 3)}%")
                else:
                    print(f"Modelo mejorado --> No cumple el patrón en un {round(self.best_features_predictions_proba[j][i][0]*100, 3)}%")

    # Realizar predicción a partir de datos externos
    def get_prediction(self):
        # Obtener los datos a predecir
        test_data = pd.read_json(Path(ml_parameters.test_data_filename))
        # Obtener predicciones y probabilidades de los datos de prueba con el modelo entrenado
        predictions_proba = self.classifier.predict_proba(test_data)
        # Realizar predicciones con los datos de prueba
        best_features_predictions_proba = self.best_features_classifier.predict_proba(test_data)

        predictions_list = [] # Lista que contiene los resultados de las predicciones para cada circuito y patrón

        # Imprimir los porcentajes de predicción
        for i in range(len(predictions_proba[0])): # Para cada circuito
            print(f"\nCircuito {i+1}:")
            circuit_dict = {} # Diccionario que almacena los resultados por patrón de una circuito
            for j in range(len(predictions_proba)): # Para cada patrón
                print(f"Patrón {ml_parameters.patterns_list[j]}: ")
                results_dict = {
                    'result': True,
                    'probability': 0.0
                }
                if len(predictions_proba[j][i]) > 1 and (predictions_proba[j][i][1] > predictions_proba[j][i][0]):
                    print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[j][i][1]*100, 2)}%")
                else:
                    print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[j][i][0]*100, 2)}%")
                
                if len(best_features_predictions_proba[j][i]) > 1 and (best_features_predictions_proba[j][i][1] > best_features_predictions_proba[j][i][0]):
                    print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[j][i][1]*100, 2)}%")
                    results_dict['probability'] = round(best_features_predictions_proba[j][i][1]*100, 2)
                else:
                    print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[j][i][0]*100, 2)}%")
                    results_dict['result'] = False
                    results_dict['probability'] = round(best_features_predictions_proba[j][i][0]*100, 2)

                circuit_dict[ml_parameters.patterns_list[j]] = results_dict

            predictions_list.append(circuit_dict)

        return predictions_list

class KNNClassifierModel(MultilabelLearningModel):
    def __init__(self, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        super().__init__(data_filename, test_size)

    def _init_model(self):
        # Modelo de ML a utilizar
        model = KNeighborsClassifier(n_neighbors=5) # Inicializado con el valor por defecto
        
        # Cuadrícula de hiperparámetros a buscar
        param_grid = {
            'classifier__n_neighbors': [1, 3, 5, 7, 9], # Mejor valores impares para evitar empates
            'classifier__weights': ['uniform', 'distance']
        }

        return model, param_grid