import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def depth_CART(x, y):
    # Utilizamos un grid para encontrar la mejor profundidad
    # REFERENCIAS: https://www.dezyre.com/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python#:~:text=DecisionTreeClassifier%20requires%20two%20parameters%20%27criterion%27%20and%20%27max_depth%27%20to,parameter.%20criterion%20%3D%20%5B%27gini%27%2C%20%27entropy%27%5D%20max_depth%20%3D%20%5B2%2C4%2C6%2C8%2C10%2C12%5D
    # StandardScaler se utiliza para eliminar los esquemas y escalar los datos haciendo que la media de los datos sea 0 y la desviación estándar como 1
    std_slc = StandardScaler()
    # Análisis de componentes principales (PCA), que reducirá la dimensión de las características mediante la creación de nuevas características que tienen la mayor parte de la variedad de los datos originales.
    pca = decomposition.PCA()
    # Elegimos CART como un modelo de aprendizaje automático para usar GridSearchCV
    dec_tree = tree.DecisionTreeClassifier()
    # Pipeline para pasar los módulos uno por uno a través de GridSearchCV para los que queremos obtener los mejores parámetros
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])
    # Ahora tenemos que definir los parámetros que queremos optimizar para estos tres objetos
    # StandardScaler no requiere que GridSearchCV optimice ningún parámetro
    # El análisis de componentes principales requiere un parámetro 'n_components' para ser optimizado. 'n_components' significa el número de componentes que se deben conservar después de reducir la dimensión
    n_components = list(range(1, x.shape[1] + 1, 1))
    # DecisionTreeClassifier requiere dos parámetros 'criterio' y 'max_depth' para ser optimizados por GridSearchCV
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12]
    # Diccionario para configurar todas las opciones de parámetros para diferentes objetos.
    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)
    # Hacer un objeto clf_GS para GridSearchCV y ajustar el conjunto de datos, es decir, x e y
    clf_GS = GridSearchCV(pipe, parameters)
    clf_GS.fit(x, y)
    # Resultados
    criterion = clf_GS.best_estimator_.get_params()['dec_tree__criterion']
    depth = clf_GS.best_estimator_.get_params()['dec_tree__max_depth']
    return criterion, depth

def show(df_mapped, x_train, y_train, x_validation, criterion, depth):
    # Realizamos predicciones con el dataset de validación
    model = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
    # Ejecutamos el modelo
    model.fit(x_train, y_train)
    # Mostramos el árbol gráficamente y guardamos una imagen
    data = tree.export_graphviz(model, out_file=None, feature_names=df_mapped.columns.values[0:6],
                                class_names=["acc", "good", "unacc", "vgood"], filled=True, rounded=True,
                                special_characters=True)
    graph = pydotplus.graph_from_dot_data(data)
    # Lo guardamos como imagen y mostramos
    graph.write_png('mydecisiontree.png')
    img = pltimg.imread('mydecisiontree.png')
    plt.imshow(img)
    plt.show()
