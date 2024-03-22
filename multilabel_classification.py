from sklearn.calibration import cross_val_predict
import utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# Leer el csv
data = pd.read_csv(Path("./datasets/dataset_openqasm_qiskit.csv"), delimiter=";")

# Mostrar la estructura de los datos
# print(data.head())
data.info()
# print(data["role"].value_counts())
print(data.describe())

# data.hist(bins=50)
# utils.save_fig("attribute_histogram_plots")
# plt.show()

# Limpiar las filas nulas
data = data.dropna()

# Obtener el conjunto de prueba
print("Nº de archivos: ", len(data))

train_set, test_set = train_test_split(data, test_size=0.3)
print("Tamaño del conjunto de entrenamiento: "+str(len(train_set))+" ("+str(round((len(train_set)/len(data))*100, 2))+"%)")
print("Tamaño del conjunto de prueba: "+str(len(test_set))+" ("+str(round((len(test_set)/len(data))*100, 2))+"%)")

# Preparar los datos para el algoritmo
# Obtener los datos sin la etiqueta
train_set_values = train_set.drop(["p.initialization", "p.superposition", "p.oracle", "p.entanglement"], axis=1)
# Almacenar las etiquetas en un diccionario
train_set_labels = dict()
train_set_labels['p.initialization'] = train_set["p.initialization"].copy()
train_set_labels['p.superposition'] = train_set["p.superposition"].copy()
train_set_labels['p.oracle'] = train_set["p.oracle"].copy()
train_set_labels['p.entanglement'] = train_set["p.entanglement"].copy()

# Realizar lo mismo para el conjunto de prueba
test_set_values = train_set.drop(["p.initialization", "p.superposition", "p.oracle", "p.entanglement"], axis=1)
# Almacenar las etiquetas en un diccionario
test_set_labels = dict()
test_set_labels['p.initialization'] = train_set["p.initialization"].copy()
test_set_labels['p.superposition'] = train_set["p.superposition"].copy()
test_set_labels['p.oracle'] = train_set["p.oracle"].copy()
test_set_labels['p.entanglement'] = train_set["p.entanglement"].copy()

# Escalar los atributos
train_set_num = train_set_values.select_dtypes(include=[np.number])
test_set_num = test_set_values.select_dtypes(include=[np.number])
scaler = StandardScaler()
train_set_num = scaler.fit_transform(train_set_num)
test_set_num = scaler.transform(test_set_num)

# Crear el clasificador multietiqueta
knn_classifier = KNeighborsClassifier()

# Entrenar el clasificador
train_set_labels_np_matrix = np.c_[
    train_set_labels['p.initialization'],
    train_set_labels['p.superposition'],
    train_set_labels['p.oracle'],
    train_set_labels['p.entanglement']]
knn_classifier.fit(train_set_num, train_set_labels_np_matrix)

# Realizar predicciones en el conjunto de prueba
test_pred = knn_classifier.predict(test_set_num)

# Evaluar el rendimiento del modelo
test_set_labels_np_matrix = np.c_[
    test_set_labels['p.initialization'],
    test_set_labels['p.superposition'],
    test_set_labels['p.oracle'],
    test_set_labels['p.entanglement']]
accuracy = accuracy_score(test_set_labels_np_matrix, test_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Evaluar la exactitud mediante evaluación cruzada
cross_val = cross_val_predict(knn_classifier, train_set_num, train_set_labels_np_matrix, cv=3)
print("Validación cruzada: ", cross_val)

# Calcular el f1 score
f1_score = f1_score(train_set_labels_np_matrix, cross_val, average="weighted", zero_division=np.nan)
print("F1 score: ", f1_score)

# Imprimir el informe de clasificación
print("Informe de clasificación:")
print(classification_report(test_set_labels_np_matrix, test_pred, zero_division=np.nan, target_names=list(train_set_labels.keys())))

# Calcular la matriz de confusión en el conjunto de prueba
conf_matrix = multilabel_confusion_matrix(test_set_labels_np_matrix, test_pred)

# Imprimir la matriz de confusión
print("Matriz de Confusión:")
print(conf_matrix)