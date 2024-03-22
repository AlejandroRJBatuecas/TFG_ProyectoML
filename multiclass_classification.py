import utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Leer el csv
data = pd.read_csv(Path("ck.csv"))

# Mostrar la estructura de los datos

# print(data.head())
# data.info()
# print(data["role"].value_counts())
print(data.describe())

# data.hist(bins=50)
# utils.save_fig("attribute_histogram_plots")
# plt.show()

# Limpiar las filas nulas
data = data.dropna()

# Obtener el conjunto de prueba
#print("Nº de archivos: ", len(data))
print("Nº de proyectos: ", len(data.groupby('project')))

train_set, test_set = utils.create_test_set(data, 0.7)
print("Tamaño del conjunto de entrenamiento: "+str(len(train_set))+" ("+str(round((len(train_set)/len(data))*100, 2))+"%)")
print("Tamaño del conjunto de prueba: "+str(len(test_set))+" ("+str(round((len(test_set)/len(data))*100, 2))+"%)")

# Preparar los datos para el algoritmo
# Obtener las etiquetas
train_set_values = train_set.drop("role", axis=1)
train_set_labels = train_set["role"].copy()
test_set_values = test_set.drop("role", axis=1)
test_set_labels = test_set["role"].copy()

# Escalar los atributos
train_set_num = train_set_values.select_dtypes(include=[np.number])
test_set_num = test_set_values.select_dtypes(include=[np.number])
scaler = StandardScaler()
train_set_num = scaler.fit_transform(train_set_num)
test_set_num = scaler.transform(test_set_num)

# Crear el clasificador
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el clasificador
classifier.fit(train_set_num, train_set_labels)

# Realizar predicciones en el conjunto de prueba
test_pred = classifier.predict(test_set_num)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(test_set_labels, test_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Evaluar la exactitud mediante evaluación cruzada
cross_val = cross_val_score(classifier, train_set_num, train_set_labels, cv=3, scoring="accuracy")
print("Validación cruzada: ", cross_val)

# Imprimir el informe de clasificación
print("Informe de clasificación:")
print(classification_report(test_set_labels, test_pred))

# Calcular la matriz de confusión en el conjunto de prueba
conf_matrix = confusion_matrix(test_set_labels, test_pred)

# Imprimir la matriz de confusión
print("Matriz de Confusión:")
print(conf_matrix)