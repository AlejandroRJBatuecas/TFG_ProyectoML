import utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SequentialFeatureSelector, RFE
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Mostrar la estructura de los datos
def show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels):
    # Mostrar los 5 primeros registros
    print(data.head())
    # Mostrar la estructura de los datos
    data.info()
    # Mostrar las estadísticas de los datos numéricos
    print(data.describe())
    # Mostrar el total de patrones detectados y tamaño de los dataframes
    print("\nTotal", data_labels["p.initialization"].value_counts())
    print("Total", data_labels["p.superposition"].value_counts())
    print("Total", data_labels["p.oracle"].value_counts())

    print("\nFormato antes del train-test split:")
    print("Formato de los valores:", data_values.shape)
    print("Formato de las etiquetas:", data_labels.shape)

    # Mostrar el tamaño de los conjuntos de entrenamiento y de prueba
    print("\nFormato después del train-test split:")
    print("Formato de los valores de entrenamiento:", train_set_values.shape)
    print("Formato de los valores de prueba:", test_set_values.shape)
    print("Formato de las etiquetas de entrenamiento:", train_set_labels.shape)
    print("Formato de las etiquetas de prueba:", test_set_labels.shape)

# Crear histograma de los atributos
def create_data_histogram(data):
    data.hist(bins=50)
    utils.save_fig("attribute_histogram_plots")
    plt.show()

# Ver la correlación entre los datos
def get_correlation_matrix(data, min_correlation_value, patterns_list):
    # Calcular correlación
    correlation_matrix = data.corr().abs()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title('Matriz de correlación')
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.tight_layout()

    # Guardar la imagen
    utils.save_fig('correlation_matrix')

    # Seleccionar las características con alta correlación con las variables objetivo
    print(f"\nCaracterísticas con alta correlación (>{min_correlation_value}):")
    for pattern in patterns_list:
        # Obtener las mejores características para cada patrón
        high_correlation_features = correlation_matrix[pattern][correlation_matrix[pattern] > min_correlation_value].index.tolist()
        if pattern in high_correlation_features:
            high_correlation_features.remove(pattern)  # Eliminar la variable objetivo de la lista
        print(f"{pattern}:", high_correlation_features)

# Obtener la información mutua respecto a las variables objetivo
# Seleccionar las mejores características con Búsqueda hacia adelante
def get_best_features(data_values, scaled_data_values, data_labels, train_set_values, train_set_labels, min_importance_values):
    best_features = dict()

    print("\nCaracterísticas seleccionadas por patrón:")
    for index, label in enumerate(data_labels):
        print("\nPatrón :", label)
        pattern = data_labels[label]
        '''
        # Información mutua
        # # Aplicar la información mutua para seleccionar las 5 mejores características
        selector = SelectKBest(mutual_info_regression, k=5)
        selector.fit_transform(data_values, pattern)
        index_list = selector.get_support(indices=True)
        features_names = [data_values.columns[index] for index in index_list]
        print(f"Información mutua:", features_names)
        #best_features_set = set(features_names)

        # Búsqueda hacia adelante
        # # Inicializar el modelo de regresión lineal
        lineal_regressor = LinearRegression()
        # # Aplicar la Búsqueda hacia adelante
        sequential_feature_selector = SequentialFeatureSelector(lineal_regressor, n_features_to_select=5, direction='forward')
        sequential_feature_selector.fit(train_set_values, train_set_labels[label])
        # Identificar las características seleccionadas
        index_list = sequential_feature_selector.get_support(indices=True)
        features_names = [data_values.columns[index] for index in index_list]
        print("Búsqueda hacia adelante:", features_names)
        #best_features_set = best_features_set.union(set(features_names))

        # Eliminación Recursiva de Características (RFE)
        # # Inicializar el modelo de regresión lineal
        lineal_regressor = LinearRegression()
        # # Aplicar RFE
        selector = RFE(lineal_regressor, n_features_to_select=5)
        selector = selector.fit(data_values, pattern)
        # # Identificar las características seleccionadas
        selected_features = [data_values.columns[index] for index in range(len(selector.support_)) if selector.support_[index]]
        print("RFE:", selected_features)
        #best_features_set = best_features_set.union(set(selected_features))
        #best_features[label] = list(best_features_set)

        # Método de regularización L1 - Lasso
        # # Normalizar las características
        if label != "p.entanglement": # Da fallo para este patrón, ya que no hay ningún registro que lo implemente
            # # Ajustar el modelo Lasso con validación cruzada para encontrar el mejor alpha
            lasso = LassoCV(cv=5, max_iter=10000)
            lasso.fit(scaled_data_values, pattern)

            # # Identificar las características seleccionadas (coeficientes diferentes de cero)
            index_list = np.where(lasso.coef_ != 0)[0]
            selected_features = [data_values.columns[index] for index in index_list]
            print("Lasso:", selected_features)
        '''
        # Evaluación de la importancia de las características - Random Forest
        # # Ajustar el modelo de Random Forest
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(scaled_data_values, pattern)

        # Obtener la importancia de las características
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        pattern_best_features = []
        i = 0
        seguir = True
        # # Imprimir la importancia de las características
        print("Importancia de las características:")
        while i < data_values.shape[1] and seguir:
            feature = data_values.columns[indices[i]]
            feature_importance = importances[indices[i]]
            print(f"{feature}: {feature_importance}")
            # Añadir a la lista de mejores características si su importancia es mayor del valor mínimo
            if feature_importance > min_importance_values[index]:
                pattern_best_features.append(feature)
            i += 1

        # Añadir la lista de características del patrón a la lista general
        best_features[label] = pattern_best_features
        
    return best_features

# Evaluar el rendimiento del modelo
def model_performance_data(data_labels_np_matrix, predictions, patterns_list):
    # Evaluar la exactitud
    accuracy = accuracy_score(data_labels_np_matrix, predictions)
    print(f"Exactitud: {round(accuracy*100, 2)}%")

    # Evaluar la precisión
    precision = precision_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)
    print(f"Precisión: {round(precision*100, 2)}%")

    # Evaluar la sensibilidad
    recall = recall_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)
    print(f"Sensibilidad: {round(recall*100, 2)}%")

    # Calcular el f1 score
    f1 = f1_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)
    print(f"F1 score: {round(f1*100, 2)}%")
    
    if len(data_labels_np_matrix.shape) > 1:
        # Imprimir el informe de clasificación
        print("Informe de clasificación:")
        print(classification_report(data_labels_np_matrix, predictions, target_names=patterns_list, zero_division=np.nan))

        # Imprimir la matriz de confusión
        print("Matriz de Confusión:")
        print(multilabel_confusion_matrix(data_labels_np_matrix, predictions))
    else:
        # Imprimir la matriz de confusión
        print("Matriz de Confusión:")
        print(confusion_matrix(data_labels_np_matrix, predictions), "\n", confusion_matrix(data_labels_np_matrix, predictions, normalize='true'))