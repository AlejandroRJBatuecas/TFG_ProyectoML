import utils
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Leer el csv
data = pd.read_csv(Path("ck.csv"))

# Mostrar la estructura de los datos
'''
print(data.head())
data.info()
print(data["role"].value_counts())
print(data.describe())

data.hist(bins=50)
utils.save_fig("attribute_histogram_plots")
plt.show()
'''

# Limpiar las filas nulas
data = data.dropna()

# Obtener el conjunto de prueba
#print("Nº de archivos: ", len(data))
print("Nº de proyectos: ", len(data.groupby('project')))

train_set, test_set = utils.create_test_set(data, 0.7)
print("Tamaño del conjunto de entrenamiento: ", len(train_set))
print("Tamaño del conjunto de prueba: ", len(test_set))