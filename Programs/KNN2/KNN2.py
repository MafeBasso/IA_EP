import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from statistics import mean


def read_dataset():
    return pandas.read_csv("Dataset_1.csv")


separator = "--------------------------------------------------------------------------"


def print_dataset(name):
    print(separator)
    print(name)
    print()
    print(dataset)
    print(separator)
    print()


def pre_processing():
    # Como apenas a coluna target possui strings, ela recebe um tratamento diferente
    for column in dataset.columns:
        if column == "target":
            mode = dataset[column].mode()
            for i in range(len(dataset)):
                if dataset[column].at[i] == "":
                    dataset[column].at[i] = mode
        else:
            mean = dataset[column].mean()
            for i in range(len(dataset)):
                if dataset[column].at[i] == 0.0:
                    dataset[column].at[i] = mean
    return dataset


def normalize():
    # Removemos a coluna target para fazer a normalização porque ela possui apenas strings
    normalized = dataset.drop("target", 1)
    normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min()))
    normalized = normalized.join(dataset["target"])
    return normalized


def correlation_matrix():
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlação entre variáveis do dataset")
    plt.show()


def exploratory_analysis():
    print(separator)
    print("Análise Exploratória")
    print()
    for column in dataset.columns:
        # Como apenas a coluna target possui strings, ela recebe um tratamento diferente
        if (column == "target"):
            mode = dataset[column].mode()
            print("Atributo: " + column)
            print("Mode: " + mode[0])
        else:
            print("Atributo: " + column)
            print("Mean: " + str(dataset[column].mean()))
            print("Min: " + str(dataset[column].min()))
            print("Max: " + str(dataset[column].max()))
            print("Median: " + str(dataset[column].median()))
            print("Var: " + str(dataset[column].var()))
            print()
    print(separator)


def knn(k, p, X_train, X_test, Y_train, Y_test):
    
    #n_neighbors é o k
    #p = 1 é equivalente à distância de manhattan
    #p = 2 é equivalente à distância euclidiana
    #p > 2 é a distância de minkowski
    KNN = KNeighborsClassifier(n_neighbors=k, p=p)

    #treinamento
    KNN.fit(X_train,Y_train)

    #predição
    Y_pred = KNN.predict(X_test)

    #comparação entre o predito e o real
    print(classification_report(Y_test, Y_pred))
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average=None)
    precision2 = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average=None)
    recall2 = recall_score(Y_test, Y_pred, average='macro')
    F_measure = f1_score(Y_test, Y_pred, average=None)
    F_measure2 = f1_score(Y_test, Y_pred, average='macro')
    return accuracy, precision, precision2, recall, recall2, F_measure, F_measure2


def stratifiedKFoldCrossValidation_with_KNN(k, p):

    print("Teste com k=" + str(k) + " e p=" + str(p) + ":\n")

    X = dataset.drop('target', axis=1)
    Y = dataset.target

    accuracies = []
    precisions = []
    precisions2 = []
    recalls = []
    recalls2 = []
    F_measures = []
    F_measures2 = []

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, Y)

    r = 1
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        print("KNN " + str(r) + " de 10:")
        acc, prec, prec2, rec, rec2, f, f2 = knn(k, p, X_train, X_test, Y_train, Y_test)

        accuracies.append(acc)
        precisions.append(prec)
        precisions2.append(prec2)
        recalls.append(rec)
        recalls2.append(rec2)
        F_measures.append(f)
        F_measures2.append(f2)

        r += 1
    
    print("Media da acuracia:", mean(accuracies))
    print("Media macro da precisao:", mean(precisions2))
    print("Media macro da sensibilidade:", mean(recalls2))
    print("Media macro da f-measure:", mean(F_measures2))
    print("Media da precisao de Bus, Car, Still, Train, Walking:", np.sum(precisions, 0)/len(precisions))
    print("Media da sensibilidade de Bus, Car, Still, Train, Walking:", np.sum(recalls, 0)/len(recalls))
    print("Media da f-measure de Bus, Car, Still, Train, Walking:", np.sum(F_measures, 0)/len(F_measures))
    print()

    plt.figure(figsize=(10,6))
    plt.title("Medias do teste com k=" + str(k) + " e p=" + str(p))
    plt.plot("acuracia", mean(accuracies), 'r.', label='acuracia')
    plt.plot("precisao\nmacro", mean(precisions2), 'g.', label='precisao macro')
    plt.plot("sensibilidade\nmacro", mean(recalls2), 'm.', label='sensibilidade macro')
    plt.plot("f-measure\nmacro", mean(F_measures2), 'y.', label='f-measure macro')
    plt.plot(["Bus", "Car", "Still", "Train", "Walking"], np.sum(precisions, 0)/len(precisions), 'c.-', label='precisao por target')
    plt.plot(["Bus", "Car", "Still", "Train", "Walking"], np.sum(recalls, 0)/len(recalls), 'b.-', label='sensibilidade por target')
    plt.plot(["Bus", "Car", "Still", "Train", "Walking"], np.sum(F_measures, 0)/len(F_measures), 'k.-', label='f-measure por target')
    plt.legend()
    plt.show()

    return mean(accuracies), mean(precisions2), mean(recalls2), mean(F_measures2)
    

def comparacao_final(lista):
    support_line_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10,6))
    plt.title("Medias de todos os testes:")
    j = 0
    for i in range(len(lista)):
        plt.plot(["acuracia", "precisao\nmacro", "sensibilidade\nmacro", "f-measure\nmacro"], lista[i], str(support_line_colors[j])+'.-', label='teste '+str(i+1))
        if(j >= len(support_line_colors)): j = -1
        j += 1
    plt.legend()
    plt.show()


# Lê e imprime o dataset original
dataset = read_dataset()

# Faz o pré-processamento dos dados
dataset = pre_processing()

# Normaliza o dataset usando a re-escala linear e imprime ele
dataset = normalize()

# Mostra a análise exploratória dos dados
#exploratory_analysis()
#correlation_matrix()

#testes
lista = []
lista.append(stratifiedKFoldCrossValidation_with_KNN(k=1, p=2))
lista.append(stratifiedKFoldCrossValidation_with_KNN(k=1, p=1))
lista.append(stratifiedKFoldCrossValidation_with_KNN(k=1, p=3))
lista.append(stratifiedKFoldCrossValidation_with_KNN(k=5, p=1))
lista.append(stratifiedKFoldCrossValidation_with_KNN(k=10, p=2))
comparacao_final(lista)