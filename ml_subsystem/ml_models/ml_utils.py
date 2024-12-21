import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from .utils import save_fig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix

# Mostrar la estructura de los datos
def show_data_structure(data, data_values, data_labels, train_set_values, test_set_values, train_set_labels, test_set_labels):
    # Mostrar los 5 primeros registros
    print(data.head())
    # Mostrar la estructura de los datos
    data.info()
    # Mostrar las estadísticas de los datos numéricos
    print(data.describe())
    # Mostrar el total de patrones detectados y tamaño de los dataframes
    for pattern in data_labels.columns:
        print("\nTotal", data_labels[pattern].value_counts())

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
    save_fig("attribute_histogram_plots")
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
    save_fig('correlation_matrix')

    # Seleccionar las características con alta correlación con las variables objetivo
    print(f"\nCaracterísticas con alta correlación (>{min_correlation_value}):")
    for pattern in patterns_list:
        # Obtener las mejores características para cada patrón
        high_correlation_features = correlation_matrix[pattern][correlation_matrix[pattern] > min_correlation_value].index.tolist()
        if pattern in high_correlation_features:
            high_correlation_features.remove(pattern)  # Eliminar la variable objetivo de la lista
        print(f"{pattern}:", high_correlation_features)

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