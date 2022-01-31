from matplotlib import pyplot


def txt_graph(df):
    # Datos del dataset (formato: texto)
    print(df)
    print(df.dtypes)

    # Datos del dataset (formato: gráfico): Diagrama de caja y bigotes
    df.plot(kind='box', layout=(6, 3))
    pyplot.xticks([1, 2, 3, 4, 5, 6],
                  ['Precio del Coche', 'Precio del Mantenimiento', 'Número de Puertas', 'Numero de Plazas',
                   'Tamaño del maletero', 'Seguridad'], size='large',
                  color='k')  # Colocamos las etiquetas para cada distribución
    pyplot.ylabel(u'Valores')
    pyplot.show()


def answers(df_mapped):
    # Número de clases, indicando que representan dichas clases y el tipo de valor que toman.
    # Número de instancias en total
    # Número de instancias pertenecientes a cada clase.
    print(df_mapped['class'].value_counts())
    print(df_mapped['class'].value_counts().sum())
    # Número de atributos de entrada, su significado y tipo.
    print(df_mapped['Buying'].value_counts())
    print(df_mapped['Buying'].value_counts().sum())
    print(df_mapped['Maintenance'].value_counts())
    print(df_mapped['Maintenance'].value_counts().sum())
    print(df_mapped['Doors'].value_counts())
    print(df_mapped['Doors'].value_counts().sum())
    print(df_mapped['Person'].value_counts())
    print(df_mapped['Person'].value_counts().sum())
    print(df_mapped['lug_boot'].value_counts())
    print(df_mapped['lug_boot'].value_counts().sum())
    print(df_mapped['safety'].value_counts())
    print(df_mapped['safety'].value_counts().sum())
    # ¿Hay algún valor de atributo desconocido?
    print(df_mapped.isnull().sum())
