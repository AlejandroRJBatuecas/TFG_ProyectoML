import utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SequentialFeatureSelector, RFE
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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
def get_best_features(data_values, data_labels, train_set_values, train_set_labels, min_importance_value):
    best_features = []

    print("\nCaracterísticas seleccionadas por patrón:")
    for label in data_labels:
        print("\nPatrón :", label)

        # Información mutua
        pattern = data_labels[label]
        # # Aplicar la información mutua para seleccionar las 5 mejores características
        selector = SelectKBest(mutual_info_regression, k=5)
        selector.fit_transform(data_values, pattern)
        index_list = selector.get_support(indices=True)
        features_names = [data_values.columns[index] for index in index_list]
        print(f"Información mutua:", features_names)

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

        # Eliminación Recursiva de Características (RFE)
        # # Inicializar el modelo de regresión lineal
        lineal_regressor = LinearRegression()
        # # Aplicar RFE
        selector = RFE(lineal_regressor, n_features_to_select=5)
        selector = selector.fit(data_values, pattern)
        # # Identificar las características seleccionadas
        selected_features = [data_values.columns[index] for index in range(len(selector.support_)) if selector.support_[index]]
        print("RFE:", selected_features)

        # Método de regularización L1 - Lasso
        # # Normalizar las características
        if label != "p.entanglement": # Da fallo para este patrón, ya que no hay ningún registro que lo implemente
            scaler = StandardScaler()
            scaled_data_values = scaler.fit_transform(data_values)

            # # Ajustar el modelo Lasso con validación cruzada para encontrar el mejor alpha
            lasso = LassoCV(cv=5, max_iter=10000)
            lasso.fit(scaled_data_values, pattern)

            # # Identificar las características seleccionadas (coeficientes diferentes de cero)
            index_list = np.where(lasso.coef_ != 0)[0]
            selected_features = [data_values.columns[index] for index in index_list]
            print("Lasso:", selected_features)

        # Evaluación de la importancia de las características - Random Forest
        # Ajustar el modelo de Random Forest
        forest = RandomForestRegressor(n_estimators=100)
        forest.fit(data_values, pattern)

        # Obtener la importancia de las características
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Imprimir la importancia de las características
        print("Importancia de las características:")
        for f in range(data_values.shape[1]):
            print(f"{data_values.columns[indices[f]]}: {importances[indices[f]]}")
            # Añadir a la lista de mejores características si su importancia es mayor del 10%
            if importances[indices[f]] > min_importance_value and data_values.columns[indices[f]] not in best_features:
                best_features.append(data_values.columns[indices[f]])

    return best_features

# Obtener los datos sin etiquetas y las etiquetas
def separate_data_and_labels(data_set, labels_list):
    # Obtener los datos sin la etiqueta
    data_set_values = data_set.drop(labels_list, axis=1)

    # Almacenar las etiquetas en un diccionario
    data_set_labels = dict()
    for label in labels_list:
        data_set_labels[label] = data_set[label].copy()

    return data_set_values, data_set_labels

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

    # Imprimir el informe de clasificación
    print("Informe de clasificación:")
    print(classification_report(data_labels_np_matrix, predictions, target_names=patterns_list, zero_division=np.nan))

    # Imprimir la matriz de confusión
    print("Matriz de Confusión:")
    print(multilabel_confusion_matrix(data_labels_np_matrix, predictions))