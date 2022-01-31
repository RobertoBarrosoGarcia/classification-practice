import mappping
import comp_algorithm
import tree
import show_data
from pandas import read_csv
from pandas import options
from sklearn.model_selection import train_test_split

# Modificamos el límite de columnas que se muestra por la terminal
options.display.max_columns = None

# Lectura del dataset
url = "Laboratorio_dataset_car.csv"
df = read_csv(url, sep=';')

# Mapeado del dataset (variables categóricas a numéricas)
df_mapped = mappping.map(df)

# Muestra del dataset en formato texto y gráfico
show_data.txt_graph(df_mapped)

# Solucion a varias de las cuestiones planteadas
show_data.answers(df_mapped)

# Dividimos el dataset en 80% de datos para entrenar y 20% para testear
array = df_mapped.values
x = array[:, 0:6]
y = array[:, 6]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

# Debemos encontrar un criterio y profundidad correcta para CART
criterion, depth = tree.depth_CART(x, y)

# Comparacion de algoritmos
comp_algorithm.compare(x_train, x_validation, y_train, y_validation, criterion, depth)

# Visualizar el aspecto del arbol final resultante
tree.show(df_mapped, x_train, y_train, x_validation, criterion, depth)