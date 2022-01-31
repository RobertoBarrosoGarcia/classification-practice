import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


def compare(x_train, x_validation, y_train, y_validation, criterion, depth):
    # Cargamos los algoritmos con los criterios
    models = [('RFC', RandomForestClassifier()),
              ('CART', DecisionTreeClassifier(criterion=criterion, max_depth=depth))]

    # Evaluamos cada modelo por turnos
    results = []
    names = []
    for name, model in models:
        # Utilizaremos una validación cruzada estratificada de 10 veces (k-fold) para estimar la precisión del modelo
        k_fold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        # Resultados de la precisión
        cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('\nClasificador: %s %s %f (%f)' % (name, '\nprecision media: ', cv_results.mean(), cv_results.std()))
        # Ejecutamos modelo
        model.fit(x_train, y_train)
        # Hacemos una predicción con nuestro modelo
        predictions = model.predict(x_validation)
        # Evaluamos las predicciones y obtenemos cual ha sido la precisión
        print('Precision de una prediccion: ', accuracy_score(y_validation, predictions))
        # Matriz de confusión
        print('\nMatriz de confusion: \n', confusion_matrix(y_validation, predictions))
        # El reporte de clasificación que muestra un desglose de cada clase por precisión, recuerdo,  puntuación f1 y apoyo
        print('\nReporte de clasificacion: \n', classification_report(y_validation, predictions))

        # Valores para la comparativa
        # Falsos positivos
        FP = confusion_matrix(y_validation, predictions).sum(axis=0) - np.diag(
            confusion_matrix(y_validation, predictions))
        # Falsos negativos
        FN = confusion_matrix(y_validation, predictions).sum(axis=1) - np.diag(
            confusion_matrix(y_validation, predictions))
        # Verdaderos positivos
        TP = np.diag(confusion_matrix(y_validation, predictions))
        # Verdaderos negativos
        TN = confusion_matrix(y_validation, predictions).sum() - (FP + FN + TP)
        # True positive rate (sensitivity)
        TPR = TP / (TP + FN)
        # True negative rate (specify)
        TNR = TN / (TN + FP)
        # False positive rate
        FPR = FP / (FP + TN)
        # Positive predicted value
        PPV = TP / (TP + FP)
        # Negative predicted value
        NPV = TN / (TN + FN)

        # Mostrar Resultados
        print(name + " FP: " + str(FP.sum()))
        print(name + " FN: " + str(FN.sum()))
        print(name + " TP: " + str(TP.sum()))
        print(name + " TN: " + str(TN.sum()))
        print(name + " TPR: " + str(TPR.mean()))
        print(name + " TNR: " + str(TNR.mean()))
        print(name + " FPR: " + str(FPR.mean()))
        print(name + " PPV: " + str(PPV.mean()))
        print(name + " NPV: " + str(NPV.mean()))

    # Comparación de algortimos mediante un gráfico
    pyplot.boxplot(results, labels=names)
    pyplot.title('Comparación de algoritmos')
    pyplot.show()
