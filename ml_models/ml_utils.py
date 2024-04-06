import utils as utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score

# Limpiar números decimales
def clean_decimal_numbers(number_str):
    # Comprobar si es un float
    if isinstance(number_str, float):
        return number_str
    else:
        # Separar la parte entera y decimal
        partes = number_str.split('.', 1)
        # Eliminar puntos adicionales de la parte entera
        parte_entera_sin_puntos = partes[0].replace(".", "")
        # Reemplazar puntos en la parte decimal
        parte_decimal_con_punto = partes[1].replace(".", "")
        # Reconstruir el número con un solo punto
        numero_str_sin_puntos_extra = parte_entera_sin_puntos + "." + parte_decimal_con_punto
        # Convertir a número decimal
        numero_decimal = float(numero_str_sin_puntos_extra)
        return numero_decimal

# Limpiar los datos corruptos de las columnas - m.AvgDens - m.AvgCNOT - m.AvgToff
def clean_outliers(data):
    for row in range(len(data["m.AvgDens"])):
        data.iat[row, data.columns.get_loc('m.AvgDens')] = clean_decimal_numbers(data.iat[row, data.columns.get_loc('m.AvgDens')])
        data.iat[row, data.columns.get_loc('m.AvgCNOT')] = clean_decimal_numbers(data.iat[row, data.columns.get_loc('m.AvgCNOT')])
        data.iat[row, data.columns.get_loc('m.AvgToff')] = clean_decimal_numbers(data.iat[row, data.columns.get_loc('m.AvgToff')])

# Mostrar la estructura de los datos
def show_data_structure(data):
    # Mostrar los 5 primeros registros
    print(data.head())
    # Mostrar la estructura de los datos
    data.info()
    # Mostrar las estadísticas de los datos numéricos
    print(data.describe())

# Crear histograma de los atributos
def create_data_histogram(data):
    data.hist(bins=50)
    utils.save_fig("attribute_histogram_plots")
    plt.show()

# Ver la correlación entre los datos
def get_correlation_matrix(data):
    # Calcular correlación
    correlation_matrix = data.corr()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title('Matriz de correlación')
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.tight_layout()

    # Guardar la imagen
    utils.save_fig('matriz_correlacion.png')

    # Mostrar el gráfico
    plt.show()

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
def model_performance_data(test_set_labels, test_pred, knn_classifier, train_set_num, train_set_labels_np_matrix, train_set_labels):
    # Transponer las listas colocando los datos en columnas
    test_set_labels_np_matrix = np.c_[
        test_set_labels['p.initialization'],
        test_set_labels['p.superposition'],
        test_set_labels['p.oracle'],
        test_set_labels['p.entanglement']]
    
    # Evaluar la puntuación de exactitud
    accuracy = accuracy_score(test_set_labels_np_matrix, test_pred)
    print(f"Precisión del modelo: {round(accuracy*100, 2)}%")

    # Evaluar la exactitud mediante evaluación cruzada
    cross_val = cross_val_predict(knn_classifier, train_set_num, train_set_labels_np_matrix, cv=3)
    #print("Validación cruzada: ", cross_val)

    # Calcular el f1 score
    f1_score_value = f1_score(train_set_labels_np_matrix, cross_val, average="weighted", zero_division=np.nan)
    print(f"F1 score: {round(f1_score_value*100, 2)}%")

    # Imprimir el informe de clasificación
    print("Informe de clasificación:")
    print(classification_report(test_set_labels_np_matrix, test_pred, zero_division=np.nan, target_names=list(train_set_labels.keys())))

    # Calcular la matriz de confusión en el conjunto de prueba
    conf_matrix = multilabel_confusion_matrix(test_set_labels_np_matrix, test_pred)

    # Imprimir la matriz de confusión
    print("Matriz de Confusión:")
    print(conf_matrix)