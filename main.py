import pandas
import math
import numpy as np

def readDataset():
    return pandas.read_csv("Dataset_1.csv")

separator = "--------------------------------------------------------------------------"

def printDataset(dataset, name):
    print(separator)
    print(name)
    print()
    print(dataset)
    print(separator)
    print()

def preProcessing(dataset):
    # Como apenas a coluna target possui strings, ela recebe um tratamento diferente
    for column in dataset.columns:
        if (column == "target"):
            mode = dataset[column].mode()
            for i in range(len(dataset)):
                if (dataset[column].at[i] == ""):
                    dataset[column].at[i] = mode
        else:
            mean = dataset[column].mean()
            for i in range(len(dataset)):
                if (dataset[column].at[i] == 0.0):
                    dataset[column].at[i] = mean
    return dataset

def normalize(dataset):
    # Removemos a coluna target para fazer a normalização porque ela possui apenas strings
    normalized = dataset.drop("target", 1)
    normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min()))
    normalized = normalized.join(dataset["target"])
    return normalized

def exploratoryAnalysis(dataset):
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

def knn(dataset, query, k=1):
    if k >= len(dataset):
        k = len(dataset) - 1
    columns = dataset.columns.size
    distances = dataset.apply(lambda row: function(query, row, columns), axis=1)
    minimal_distances = distances.sort_values()[0:k]
    classes = list()
    for i in range(minimal_distances.size):
        classes.append(dataset.loc[minimal_distances.index[i]][columns - 1])
    unique_classes = np.unique(classes)
    votes_per_class = [0] * unique_classes.size
    for i in range(unique_classes.size):
        votes_per_class[i] = classes.count(unique_classes[i])
    return unique_classes[votes_per_class.index(max(votes_per_class))]

def function(query, row, classId):
    return math.sqrt(sum((query - row[0:(classId - 1)]) ** 2))

# Lê e imprime o dataset original
dataset = readDataset()
printDataset(dataset, "Dataset")

# Faz o pré-processamento dos dados
dataset = preProcessing(dataset)

# Normaliza o dataset usando a re-escala linear e imprime ele
dataset = normalize(dataset)
printDataset(dataset, "Dataset Normalizado")

# Mostra a análise exploratória dos dados
exploratoryAnalysis(dataset)

query = [78.0, 9.81147629156, 9.75889547269, 9.84941095639, 0.0146255073331, 0.00165077522136, 0.0,
         0.003533082558530001, 0.000736963641413, 0.0, 0.0, 0.0, 0.0]
print(knn(dataset, query, 10))