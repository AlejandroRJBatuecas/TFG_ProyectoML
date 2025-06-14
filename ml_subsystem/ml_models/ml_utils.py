import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import os

from .utils import save_fig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix

# Mostrar la estructura de los datos
def show_data_structure(data):
    # Mostrar la forma del dataset
    print(data.shape)
    # Mostrar la estructura de los datos
    data.info()
    # Mostrar los 5 primeros registros
    print(data.head())
    # Mostrar las estadísticas de los datos numéricos
    print(data.describe())

# Mostrar la estructura de los datos particionados
def show_split_data_structure(data_values, data_labels):
    # Mostrar el total de patrones detectados y tamaño de los dataframes
    print("\nFormato tras de la división en conjuntos de entrenamiento y prueba:")
    print("Formato de las características:", data_values.shape)
    print("Formato de las etiquetas:", data_labels.shape)

    # Crear un DataFrame con los conteos de valores True/False para cada patrón
    pattern_values_count = pd.DataFrame({
        "Patrón": data_labels.columns,
        "False": [data_labels[col].value_counts().get(False, 0) for col in data_labels.columns],
        "True": [data_labels[col].value_counts().get(True, 0) for col in data_labels.columns]
    })

    print("\nRecuento de valores por patrón:")
    print(pattern_values_count)

# Crear histograma de las características de entrada
def create_data_histogram(data, data_values):
    # Crear los histogramas de la distribución de los datos
    plt.figure(figsize=(9, 12))
    for i, column in enumerate(data_values.columns):
        plt.subplot(9, 3, i+1)
        sns.histplot(data[column], kde=True)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(f'{column}')
    plt.tight_layout()
    save_fig("attribute_histogram_plots")
    #plt.show()

# Obtener la matriz de correlación
def get_correlation_matrix(data):
    # Calcular la matriz de correlación en valor absoluto
    correlation_matrix = data.corr().abs()

    # Visualizar la matriz de correlación utilizando un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    save_fig('correlation_matrix')
    #plt.show()

    return correlation_matrix

# Analizar la correlación entre las variables
def correlation_analysis(correlation_matrix, min_correlation_value, patterns_list):
    # Crear las parejas de correlación entre las variables
    high_correlation_pairs = (
        correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    # Renombrar columnas para mayor claridad
    high_correlation_pairs.columns = ["Característica 1", "Característica 2", "Correlación"]

    # Filtrar solo las correlaciones mayores al umbral de correlación
    high_correlation_pairs = high_correlation_pairs[high_correlation_pairs["Correlación"] > min_correlation_value]

    # Ordenar de mayor a menor correlación
    high_correlation_pairs = high_correlation_pairs.sort_values(by="Correlación", ascending=False)

    # Mostrar los pares de variables con alta correlación
    print(f"\n{high_correlation_pairs}")
    
    # Seleccionar las características con alta correlación con las variables objetivo
    print(f"\nCaracterísticas con alta correlación (>{min_correlation_value}):")
    for pattern in patterns_list:
        # Obtener las mejores características para cada patrón
        high_correlation_features = correlation_matrix[pattern][correlation_matrix[pattern] > min_correlation_value].index.tolist()
        if pattern in high_correlation_features:
            high_correlation_features.remove(pattern)  # Eliminar la variable objetivo de la lista
        print(f"{pattern}:", high_correlation_features)

# Mostrar la estructura de los conjuntos de entrenamiento y prueba
def show_datasets_structure(train_set_values, test_set_values, train_set_labels, test_set_labels):
    # Crear un DataFrame con los tamaños de cada conjunto
    datasets_sizes = pd.DataFrame({
        "Dataset": ["Entrenamiento", "Entrenamiento", "Prueba", "Prueba"],
        "Datos": ["Características", "Etiquetas", "Características", "Etiquetas"],
        "Registros": [train_set_values.shape[0], train_set_labels.shape[0], test_set_values.shape[0], test_set_labels.shape[0]],
        "Columnas": [train_set_values.shape[1], train_set_labels.shape[1], test_set_values.shape[1], test_set_labels.shape[1]]
    })

    print("\nTamaños de los conjuntos de entrenamiento y prueba:")
    print(datasets_sizes)

# Normalizar la matriz de confusión multietiqueta
def normalize_confusion_matrix(mcm):
    normalized_mcm = []
    for cm in mcm:
        cm_sum = cm.astype(float).sum(axis=1, keepdims=True)
        # Evitar la división por cero
        cm_sum[cm_sum == 0] = 1
        normalized_cm = cm.astype(float) / cm_sum
        normalized_mcm.append(normalized_cm)
    return np.array(normalized_mcm)

# Evaluar el rendimiento del modelo
def model_performance_data(data_labels_np_matrix, predictions, patterns_list):
    # Evaluar la exactitud
    accuracy = round(accuracy_score(data_labels_np_matrix, predictions)*100, 3)
    print(f"Exactitud: {accuracy}%")

    # Evaluar la precisión = VP / (VP + FP), donde VP es (2,2) y FP es (1,2) de la matriz de confusión
    precision = round(precision_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)*100, 3)
    print(f"Precisión: {precision}%")

    # Evaluar la sensibilidad = VP / (VP + FN), donde VP es (2,2) y FP es (2,1) de la matriz de confusión
    recall = round(recall_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)*100, 3)
    print(f"Sensibilidad: {recall}%")

    # Calcular el f1 score (media armónica entre precision y recall) = 2 / (1/Precision + 1/Recall)
    f1 = round(f1_score(data_labels_np_matrix, predictions, average="weighted", zero_division=np.nan)*100, 3)
    print(f"F1 score: {f1}%")

    confusion_matrix_dict = {}
    normalized_confusion_matrix_dict = {}

    if len(data_labels_np_matrix.shape) > 1:
        classification_report_dict = classification_report(data_labels_np_matrix, predictions, target_names=patterns_list, zero_division=np.nan, output_dict=True)
        # Imprimir el informe de clasificación
        print("Informe de clasificación:\n", classification_report_dict)
        
        # Imprimir las matrices de confusión
        confusion_matrix_ndarray = multilabel_confusion_matrix(data_labels_np_matrix, predictions)
        normalized_confusion_matrix_ndarray = np.round(normalize_confusion_matrix(confusion_matrix_ndarray), 3)
        for i in range(len(confusion_matrix_ndarray)):
            print("Matriz de Confusión:", patterns_list[i], "\n", confusion_matrix_ndarray[i])
            print("Normalizada:\n", normalized_confusion_matrix_ndarray[i])

        # Obtener las matrices de confusión para cada patrón
        for i, matrix in enumerate(confusion_matrix_ndarray):
            confusion_matrix_dict[patterns_list[i]] = matrix

        # Obtener las matrices de confusión normalizadas para cada patrón
        for i, matrix in enumerate(normalized_confusion_matrix_ndarray):
            normalized_confusion_matrix_dict[patterns_list[i]] = matrix

    else:
        classification_report_dict = classification_report(data_labels_np_matrix, predictions, target_names=None, zero_division=np.nan, output_dict=True)
        # Imprimir el informe de clasificación
        print("Informe de clasificación:\n", classification_report_dict)
        
        # Imprimir las matrices de confusión
        confusion_matrix_ndarray = confusion_matrix(data_labels_np_matrix, predictions)
        normalized_confusion_matrix_ndarray = np.round(confusion_matrix(data_labels_np_matrix, predictions, normalize='true'), 3)
        print("Matriz de Confusión:\n", confusion_matrix_ndarray)
        print("Normalizada:\n", normalized_confusion_matrix_ndarray)

        confusion_matrix_dict = confusion_matrix_ndarray
        normalized_confusion_matrix_dict = normalized_confusion_matrix_ndarray

    model_performance_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Classification report": classification_report_dict,
        "Confusion matrix": confusion_matrix_dict,
        "Normalized confusion matrix": normalized_confusion_matrix_dict
    }

    return model_performance_dict

def store_model(ml_model, trained_model_path):
    # Separar el directorio y el nombre del archivo
    file_folder, _ = os.path.split(trained_model_path)

    # Crear el directorio si no existe
    os.makedirs(file_folder, exist_ok=True)

    # Almacenar los modelos entrenados en un archivo
    joblib.dump(ml_model, trained_model_path)