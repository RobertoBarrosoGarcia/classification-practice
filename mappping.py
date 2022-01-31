from copy import copy


def map(df):
    # Definicion del mapeo
    buying_dict = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
    maintenance_dict = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
    doors_dict = {'2': 2, '3': 3, '4': 4, '5more': 5}
    person_dict = {'2': 2, '4': 4, 'more': 5}
    lug_boot_dict = {'small': 1, 'med': 2, 'big': 3}
    safety_dict = {'low': 1, 'med': 2, 'high': 3}

    # Copia del dataframe para no pisar los datos originales
    df_mapped = copy(df)

    # Mapeado de los datos categóricos
    df_mapped['Buying'] = df_mapped['Buying'].map(buying_dict)
    df_mapped['Maintenance'] = df_mapped['Maintenance'].map(maintenance_dict)
    df_mapped['Doors'] = df_mapped['Doors'].map(doors_dict)
    df_mapped['Person'] = df_mapped['Person'].map(person_dict)
    df_mapped['lug_boot'] = df_mapped['lug_boot'].map(lug_boot_dict)
    df_mapped['safety'] = df_mapped['safety'].map(safety_dict)

    return df_mapped
