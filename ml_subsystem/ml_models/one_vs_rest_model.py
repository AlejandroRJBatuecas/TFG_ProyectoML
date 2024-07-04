import pandas as pd
import numpy as np

from config import ml_parameters
from .ml_utils import show_data_structure, get_correlation_matrix, model_performance_data
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.calibration import cross_val_predict

class OneVsRestModel:
    def __init__(self, model, param_grid, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Inicialización de parámetros
        self.model = model
        self.param_grid = param_grid
        # Leer el csv con los datos de entrenamiento
        self.data = pd.read_csv(Path(data_filename), delimiter=";")
        self.test_size = test_size

        self.data_values, self.data_labels = self._prepare_data()
        self.train_set_values, self.test_set_values, self.train_set_labels, self.test_set_labels = self._get_datasets()
        self.pipelines, self.best_features_pipelines = self._create_pipelines()
        self.classifiers, self.best_features_classifiers = self._get_classifiers()
        self.best_features = self._get_feature_importance()
        self._evaluate_model_performance()
        self._evaluate_best_features_model_performance()
        self.predictions_proba = self._test_model_performance()
        self.best_features_predictions_proba = self._test_best_features_model_performance()
        self._show_first_test_predictions()

    def _prepare_data(self):
        # Limpiar las filas con algún dato nulo
        self.data = self.data.dropna()

        # Elimnar las columnas relacionadas con Oracle
        self.data = self.data.drop(columns=ml_parameters.eliminated_metrics)

        # Separar datos y etiquetas
        data_values = self.data.select_dtypes(include=[np.number])
        data_labels = self.data[ml_parameters.patterns_list]

        return data_values, data_labels

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
        pipelines = {}
        best_features_pipelines = {}

        print(f"\nHiperparámetros por defecto de {self.model.__class__.__name__}:\n", 
                self.model.get_params())

        for i, pattern in enumerate(ml_parameters.patterns_list):
            # Crear el pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', self.model)
            ], memory=None)

            best_features_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=ml_parameters.random_state_value))),
                ('classifier', self.model)
            ], memory=None)

            # Asignar el umbral de importancia al selector de características
            best_features_pipeline.steps[1][1].threshold = ml_parameters.min_importance_values[i]

            pipelines[pattern] = pipeline
            best_features_pipelines[pattern] = best_features_pipeline

        return pipelines, best_features_pipelines
    
    # Obtener los clasificadores con los mejores hiperparámetros
    def _get_classifiers(self):
        classifiers = {}
        best_features_classifiers = {}
        
        for pattern in ml_parameters.patterns_list:
            # Obtener las etiquetas del patrón
            pattern_train_labels = self.train_set_labels[pattern]

            # Realizar la búsqueda de hiperparámetros utilizando validación cruzada
            grid_search = GridSearchCV(self.pipelines[pattern], self.param_grid, cv=ml_parameters.cv_value)
            grid_search.fit(self.train_set_values, pattern_train_labels)
            print(f"\nMejores hiperparámetros para el clasificador Normal para la etiqueta {pattern}: \n{grid_search.best_params_}")

            best_features_grid_search = GridSearchCV(self.best_features_pipelines[pattern], self.param_grid, cv=ml_parameters.cv_value)
            best_features_grid_search.fit(self.train_set_values, pattern_train_labels)
            print(f"\nMejores hiperparámetros para el clasificador Mejorado para la etiqueta {pattern}: \n{best_features_grid_search.best_params_}")

            # Obtener el mejor modelo
            classifier = grid_search.best_estimator_
            best_features_classifier = best_features_grid_search.best_estimator_

            classifiers[pattern] = classifier
            best_features_classifiers[pattern] = best_features_classifier

        return classifiers, best_features_classifiers

    # Obtener la importancia de las caracteristicas
    def _get_feature_importance(self):
        best_features = {}

        for pattern in ml_parameters.patterns_list:
            # Obtener los nombres de las características originales
            feature_names = self.data_values.columns
            # Obtener el selector de características desde la pipeline
            feature_selector = self.best_features_classifiers[pattern].named_steps['feature_selection']
            # Obtener el modelo subyacente desde el selector de características
            model = feature_selector.estimator_
            # Obtener las importancias de las características
            importances = model.feature_importances_
            # Crear un diccionario de características e importancias
            feature_importance_dict = dict(zip(feature_names, importances))
            # Ordenar las características por importancia
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            # Imprimir las importancias de las características
            print(f"\nImportancias de las características para la etiqueta {pattern}:")
            for feature, importance in sorted_features:
                print(f"{feature}: {round(importance*100, 3)}%")

            # Filtrar las características originales para obtener solo las seleccionadas
            selected_features = self.best_features_classifiers[pattern].named_steps['feature_selection'].get_support()
            pattern_best_features = [feature for feature, selected in zip(feature_names, selected_features) if selected]

            best_features[pattern] = pattern_best_features

        return best_features
    
    # Evaluar el rendimiento del modelo
    def _evaluate_model_performance(self):
        for pattern in ml_parameters.patterns_list:
            # Obtener las etiquetas del patrón
            pattern_train_labels = self.train_set_labels[pattern]
            # Calcular las predicciones del clasificador mediante evaluación cruzada
            predictions = cross_val_predict(self.classifiers[pattern], self.train_set_values, pattern_train_labels, cv=ml_parameters.cv_value)
            # Mostrar las medidas de rendimiento
            print(f"\nRendimiento del clasificador de {pattern}")
            model_performance_data(pattern_train_labels, predictions, pattern)

    # Evaluar el rendimiento del modelo con las mejores métricas
    def _evaluate_best_features_model_performance(self):
        for pattern in ml_parameters.patterns_list:
            # Obtener las etiquetas del patrón
            pattern_train_labels = self.train_set_labels[pattern]
            # Calcular las predicciones del clasificador mediante evaluación cruzada
            predictions = cross_val_predict(self.best_features_classifiers[pattern], self.train_set_values, pattern_train_labels, cv=ml_parameters.cv_value)
            # Mostrar las medidas de rendimiento
            print(f"\nMétricas seleccionadas para {pattern}: {len(self.best_features[pattern])}\n{self.best_features[pattern]}")
            print(f"\nRendimiento del modelo mejorado")
            model_performance_data(pattern_train_labels, predictions, pattern)
    
    # Evaluar el rendimiento del modelo con el conjunto de prueba
    def _test_model_performance(self):
        predictions_proba = {}

        for pattern in ml_parameters.patterns_list:
            pattern_test_labels = self.test_set_labels[pattern]
            # Realizar predicciones en el conjunto de prueba
            predictions = self.classifiers[pattern].predict(self.test_set_values)
            pattern_predictions_proba = self.classifiers[pattern].predict_proba(self.test_set_values)

            predictions_proba[pattern] = pattern_predictions_proba
            
            # Mostrar las medidas de rendimiento
            print(f"\nRendimiento del modelo de {pattern} con el conjunto de prueba")
            model_performance_data(pattern_test_labels, predictions, pattern)
            # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
            pattern_test_labels_list = pattern_test_labels.tolist()
            for i in range(len(pattern_test_labels_list)):
                if pattern_predictions_proba[i][1] > pattern_predictions_proba[i][0]:
                    if pattern_test_labels_list[i] == False:
                        print(f"\nModelo normal --> Cumple el patrón en un {round(pattern_predictions_proba[i][1]*100, 3)}%")
                        print(f"Instancia {i+1}: {pattern_test_labels_list[i]}")
                else:
                    if pattern_test_labels_list[i] == True:
                        print(f"\nModelo normal --> No cumple el patrón en un {round(pattern_predictions_proba[i][0]*100, 3)}%")
                        print(f"Instancia {i+1}: {pattern_test_labels_list[i]}")

        return predictions_proba
    
    # Evaluar el rendimiento del modelo con mejores métricas con el conjunto de prueba
    def _test_best_features_model_performance(self):
        best_features_predictions_proba = {}

        for pattern in ml_parameters.patterns_list:
            pattern_test_labels = self.test_set_labels[pattern]
            # Realizar predicciones en el conjunto de prueba
            predictions = self.best_features_classifiers[pattern].predict(self.test_set_values)
            pattern_best_features_predictions_proba = self.best_features_classifiers[pattern].predict_proba(self.test_set_values)
            
            best_features_predictions_proba[pattern] = pattern_best_features_predictions_proba
            
            # # Mostrar las medidas de rendimiento
            print(f"\nMétricas seleccionadas para {pattern}: {len(self.best_features[pattern])}\n{self.best_features[pattern]}")
            print("\nRendimiento del modelo mejorado con el conjunto de prueba")
            model_performance_data(pattern_test_labels, predictions, pattern)
            # # Imprimir los porcentajes de predicción de los registros mal predichos del conjunto de prueba
            pattern_test_labels_list = pattern_test_labels.tolist()
            for i in range(len(pattern_test_labels_list)):
                if pattern_best_features_predictions_proba[i][1] > pattern_best_features_predictions_proba[i][0]:
                    if pattern_test_labels_list[i] == False:
                        print(f"\nModelo mejorado --> Cumple el patrón en un {round(pattern_best_features_predictions_proba[i][1]*100, 3)}%")
                        print(f"Instancia {i+1}: {pattern_test_labels_list[i]}")
                else:
                    if pattern_test_labels_list[i] == True:
                        print(f"\nModelo mejorado --> No cumple el patrón en un {round(pattern_best_features_predictions_proba[i][0]*100, 3)}%")
                        print(f"Instancia {i+1}: {pattern_test_labels_list[i]}")

        return best_features_predictions_proba
    
    # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
    def _show_first_test_predictions(self, test_results_num=ml_parameters.test_results_num):
        for pattern in ml_parameters.patterns_list:
            pattern_test_labels_list = self.test_set_labels[pattern].tolist()
            print(f"\nResultados de los primeros {test_results_num} registros de prueba para la etiqueta {pattern}:")
            for i in range(0, test_results_num):
                print(f"\nInstancia {i+1}: {pattern_test_labels_list[i]}")
                if self.predictions_proba[pattern][i][1] > self.predictions_proba[pattern][i][0]:
                    print(f"Modelo normal --> Cumple el patrón en un {round(self.predictions_proba[pattern][i][1]*100, 3)}%")
                else:
                    print(f"Modelo normal --> No cumple el patrón en un {round(self.predictions_proba[pattern][i][0]*100, 3)}%")

                if self.best_features_predictions_proba[pattern][i][1] > self.best_features_predictions_proba[pattern][i][0]:
                    print(f"Modelo mejorado --> Cumple el patrón en un {round(self.best_features_predictions_proba[pattern][i][1]*100, 3)}%")
                else:
                    print(f"Modelo mejorado --> No cumple el patrón en un {round(self.best_features_predictions_proba[pattern][i][0]*100, 3)}%")

    def show_model_evaluation(self, test_results_num=ml_parameters.test_results_num):
        # Evaluar el rendimiento del modelo
        self._evaluate_model_performance()
        self._evaluate_best_features_model_performance()
        # Imprimir los porcentajes de predicción de los primeros registros del conjunto de prueba
        self._show_first_test_predictions(test_results_num)

    # Realizar predicción a partir de datos externos
    def get_prediction(self):
        # Obtener los datos a predecir
        test_data = pd.read_csv(Path(ml_parameters.test_data_filename), delimiter=";")

        for pattern in ml_parameters.patterns_list:
            # Obtener predicciones y probabilidades de los datos de prueba con el modelo entrenado
            test_data_pred = self.classifiers[pattern].predict(test_data)
            predictions_proba = self.classifiers[pattern].predict_proba(test_data)
            # Realizar predicciones con los datos de prueba
            best_features_test_data_pred = self.best_features_classifiers[pattern].predict(test_data)
            best_features_predictions_proba = self.best_features_classifiers[pattern].predict_proba(test_data)
            # Imprimir los porcentajes de predicción
            for i in range(len(predictions_proba)):
                print(f"\nInstancia {i+1}: {pattern}")
                if predictions_proba[i][1] > predictions_proba[i][0]:
                    print(f"Modelo normal --> Cumple el patrón en un {round(predictions_proba[i][1]*100, 3)}%")
                else:
                    print(f"Modelo normal --> No cumple el patrón en un {round(predictions_proba[i][0]*100, 3)}%")

                if best_features_predictions_proba[i][1] > best_features_predictions_proba[i][0]:
                    print(f"Modelo mejorado --> Cumple el patrón en un {round(best_features_predictions_proba[i][1]*100, 3)}%")
                else:
                    print(f"Modelo mejorado --> No cumple el patrón en un {round(best_features_predictions_proba[i][0]*100, 3)}%")

class KNNOvsRClassifierModel(OneVsRestModel):
    def __init__(self, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Definir la cuadrícula de hiperparámetros a buscar
        param_grid = {
            'classifier__n_neighbors': [1, 3, 5, 7, 9], # Mejor valores impares para evitar empates
            'classifier__weights': ['uniform', 'distance']
        }
        model = KNeighborsClassifier()
        super().__init__(model, param_grid, data_filename, test_size)

class RandomForestClassifierModel(OneVsRestModel):
    def __init__(self, data_filename=ml_parameters.data_filename, test_size=ml_parameters.test_set_size):
        # Parámetros para la búsqueda aleatoria de hiperparámetros
        param_grid = {
            'classifier__n_estimators': np.linspace(100, 3000, 10, dtype=int),
            'classifier__min_samples_split': [2, 3, 4, 5, 10, 20, 40, 80], # Valores [2,inf] o [0.0,1.0]
            'classifier__min_samples_leaf': [2, 3, 4, 5, 10, 20, 40, 80], # Mínimo 2 por hoja para no crear hojas específicas
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__max_depth': [10, 20, 40, 80, 160], # Valores [1,inf]
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__bootstrap': [True, False]
        }
        model = RandomForestClassifier()
        super().__init__(model, param_grid, data_filename, test_size)

    # Obtener los clasificadores con los mejores hiperparámetros
    def _get_classifiers(self):
        classifiers = {}
        best_features_classifiers = {}
        
        for pattern in ml_parameters.patterns_list:
            # Obtener las etiquetas del patrón
            pattern_train_labels = self.train_set_labels[pattern]

            # Realizar la búsqueda aleatoria de hiperparámetros
            rf_random = RandomizedSearchCV(self.pipelines[pattern], self.param_grid, n_iter=ml_parameters.n_iter_value, cv=ml_parameters.cv_value)
            rf_random.fit(self.train_set_values, pattern_train_labels)
            print("\nMejores hiperparámetros para el clasificador Normal para la etiqueta", pattern, ": \n", rf_random.best_params_)

            # Realizar la búsqueda aleatoria de hiperparámetros para el modelo con mejores características
            best_features_rf_random = RandomizedSearchCV(self.best_features_pipelines[pattern], self.param_grid, n_iter=ml_parameters.n_iter_value, cv=ml_parameters.cv_value)
            best_features_rf_random.fit(self.train_set_values, pattern_train_labels)
            print("\nMejores hiperparámetros para el clasificador Mejorado para la etiqueta", pattern, ": \n", best_features_rf_random.best_params_)

            # Obtener el mejor modelo
            classifier = rf_random.best_estimator_
            best_features_classifier = best_features_rf_random.best_estimator_

            classifiers[pattern] = classifier
            best_features_classifiers[pattern] = best_features_classifier

        return classifiers, best_features_classifiers