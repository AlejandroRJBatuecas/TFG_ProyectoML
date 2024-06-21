from sklearn.calibration import cross_val_predict
import ml_models.utils as utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Leer el csv
data = pd.read_csv(Path("../datasets/dataset_openqasm_qiskit.csv"), delimiter=";")

# Mostrar la estructura de los datos
# print(data.head())
data.info()
# print(data["role"].value_counts())
print(data.describe())

# data.hist(bins=50)
utils.save_fig("attribute_histogram_plots")
# plt.show()

# Limpiar las filas nulas
data = data.dropna()

# Convertir las etiquetas de texto a valores binarios (0/1)
data['p.initialization'] = data['p.initialization'].apply(lambda x: 1 if x == True else 0)
data['p.superposition'] = data['p.superposition'].apply(lambda x: 1 if x == True else 0)
data['p.oracle'] = data['p.oracle'].apply(lambda x: 1 if x == True else 0)
data['p.entanglement'] = data['p.entanglement'].apply(lambda x: 1 if x == True else 0)

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

# Crear los clasificadores para cada patrón
initialization_lin_reg = LinearRegression()
superposition_lin_reg = LinearRegression()
oracle_lin_reg = LinearRegression()
entanglement_lin_reg = LinearRegression()

# Entrenar los clasificadores
initialization_lin_reg.fit(train_set_num, train_set_labels['p.initialization'])
superposition_lin_reg.fit(train_set_num, train_set_labels['p.superposition'])
oracle_lin_reg.fit(train_set_num, train_set_labels['p.oracle'])
entanglement_lin_reg.fit(train_set_num, train_set_labels['p.entanglement'])

# Realizar predicciones en el conjunto de prueba
initialization_test_pred = initialization_lin_reg.predict(test_set_num)
superposition_test_pred = superposition_lin_reg.predict(test_set_num)
oracle_test_pred = oracle_lin_reg.predict(test_set_num)
entanglement_test_pred = entanglement_lin_reg.predict(test_set_num)

print(initialization_test_pred)
media = sum(initialization_test_pred)/len(initialization_test_pred)
print("Media: ", media)
print(max(initialization_test_pred))

'''
# Evaluar la exactitud mediante evaluación cruzada
cross_val = cross_val_predict(initialization_lin_reg, train_set_num, train_set_labels['p.initialization'], cv=3)
print("Validación cruzada : ", cross_val)

# Imprimir el informe de clasificación
print("Informe de clasificación:")
print(classification_report(test_set_labels_np_matrix, test_pred, zero_division=np.nan, target_names=list(train_set_labels.keys())))
'''