import utils as utils
import ml_utils as mlutils
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Parámetros generales
test_set_size = 0.3
patterns_list = ["p.initialization", "p.superposition", "p.oracle", "p.entanglement"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Leer el csv con los datos
data = pd.read_csv(Path("../datasets/dataset_openqasm_qiskit.csv"), delimiter=";")
test_data = pd.read_csv(Path("../datasets/prueba_uno_dataset_openqasm_qiskit.csv"), delimiter=";")

# Mostrar la estructura de los datos
#mlutils.clean_outliers(data)
#mlutils.show_data_structure(data)
#mlutils.create_data_histogram(data)

# Limpiar las filas con algún dato nulo
data = data.dropna()

# Obtener el conjunto de prueba
print("Nº de archivos: ", len(data))
train_set, test_set = train_test_split(data, test_size=test_set_size)
print("Tamaño del conjunto de entrenamiento: "+str(len(train_set))+" ("+str(round((len(train_set)/len(data))*100, 2))+"%)")
print("Tamaño del conjunto de prueba: "+str(len(test_set))+" ("+str(round((len(test_set)/len(data))*100, 2))+"%)")

# Preparar los datos para el algoritmo
# Conjunto de entrenamiento
train_set_values, train_set_labels = mlutils.separate_data_and_labels(train_set, patterns_list)
# Conjunto de prueba
test_set_values, test_set_labels = mlutils.separate_data_and_labels(test_set, patterns_list)

# Obtener los valores numéricos de los datos
train_set_num = train_set_values.select_dtypes(include=[np.number])
test_set_num = test_set_values.select_dtypes(include=[np.number])

# Escalar los atributos
scaler = StandardScaler()
train_set_num = scaler.fit_transform(train_set_num)
test_set_num = scaler.transform(test_set_num)

# Transponer las listas colocando los datos en columnas
train_set_labels_np_matrix = np.c_[
    train_set_labels['p.initialization'],
    train_set_labels['p.superposition'],
    train_set_labels['p.oracle'],
    train_set_labels['p.entanglement']]

# Crear el clasificador multietiqueta
knn_classifier = KNeighborsClassifier()

# Entrenar el clasificador
knn_classifier.fit(train_set_num, train_set_labels_np_matrix)

# Realizar predicciones en el conjunto de prueba
test_pred = knn_classifier.predict(test_set_num)

# Evaluar el rendimiento del modelo
mlutils.model_performance_data(test_set_labels, test_pred, knn_classifier, train_set_num, train_set_labels_np_matrix, train_set_labels)

# Preparar los datos de prueba
test_set_data = test_data.drop(["m.AvgDens", "m.AvgCNOT", "m.AvgToff"], axis=1)
test_set_data_values = test_set_data.select_dtypes(include=[np.number])
test_set_data_values = scaler.transform(test_set_data_values)

# Obtener predicciones de los datos de prueba con el modelo entrenado
test_data_pred = knn_classifier.predict(test_set_data_values)
predictions_proba = knn_classifier.predict_proba(test_set_data_values)

# Imprimir los porcentajes de predicción
for i in range(len(predictions_proba[0])):
    print(f"Instancia {i+1}:")
    for j in range(len(predictions_proba)):
        if len(predictions_proba[j][i]) > 1 and (predictions_proba[j][i][1] > predictions_proba[j][i][0]):
            print(f" Cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][1]*100, 2)}%")
        else:
            print(f" No cumple el patrón {patterns_list[j]} en un {round(predictions_proba[j][i][0]*100, 2)}%")