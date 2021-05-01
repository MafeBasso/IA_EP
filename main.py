import random
from unicodedata import numeric

import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
from collections import Counter



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


def correlation_matrix(dataset):
    sns.heatmap(dataset.corr(), annot=True, fmt='.2f', cmap='Blues')
    plt.title('Correlação entre variáveis do dataset')
    plt.show()

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
    correlation_matrix(dataset)


def minkowski_distance(query, row, columns, p):
    subtracao = query - row[0:(columns - 1)]
    pow1 = subtracao ** p
    soma = sum(pow1)
    pow2 = soma ** (1 / p)
    return pow2


def euclidian_distance(query, row, columns):
    return minkowski_distance(query, row, columns, 2)


def manhattan_distance(query, row, columns):
    return minkowski_distance(query, row, columns, 1)


distance_types = ["minkowski", "euclidian", "manhattan", "cos"]


def distance_calculation(distance_type, dataset, query, columns, p):
    distance = 0
    if distance_type == distance_types[0]:
        distance = dataset.apply(lambda row: minkowski_distance(query, row, columns, p), axis=1)
    elif distance_type == distance_types[1]:
        distance = dataset.apply(lambda row: euclidian_distance(query, row, columns), axis=1)
    elif distance_type == distance_types[2]:
        distance = dataset.apply(lambda row: manhattan_distance(query, row, columns), axis=1)
    return distance


def remove_irrelevant_columns(dataset, query):
    dataset = dataset.drop("time", 1)
    dataset = dataset.drop("sound#mean", 1)
    dataset = dataset.drop("sound#min", 1)
    dataset = dataset.drop("sound#max", 1)
    dataset = dataset.drop("sound#std", 1)
    query.pop(12)
    query.pop(11)
    query.pop(10)
    query.pop(9)
    query.pop(0)
    return dataset, query


def printing(text, variable, knn):
    print("Linha: " + text + ": " + str(variable) + " - Resultado: " + knn)

def knn(dataset, query, distance_type, k = 1, p = 2):
    dataset, query = remove_irrelevant_columns(dataset, query)

    if k >= len(dataset):
        k = len(dataset) - 1

    columns = dataset.columns.size

    distances = distance_calculation(distance_type, dataset, query, columns, p)

    minimal_distances = distances.sort_values()[0:k]

    classes = list()

    for i in range(minimal_distances.size):
        indice_dataset = minimal_distances.index[i]
        linha_dataset = dataset.loc[indice_dataset]
        classe_escolhida = linha_dataset[columns - 1]
        classes.append(classe_escolhida)

    unique_classes = np.unique(classes)

    votes_per_class = [0] * unique_classes.size
    for i in range(unique_classes.size):
        votes_per_class[i] = classes.count(unique_classes[i])

    return unique_classes[votes_per_class.index(max(votes_per_class))]


# Lê e imprime o dataset original
dataset = readDataset()
# printDataset(dataset, "Dataset")

# Faz o pré-processamento dos dados
dataset = preProcessing(dataset)

# Normaliza o dataset usando a re-escala linear e imprime ele
dataset = normalize(dataset)
# printDataset(dataset, "Dataset Normalizado")

# Mostra a análise exploratória dos dados
# exploratoryAnalysis(dataset)

def test(total_tests, k, choosen_distance, p = 0):
    hits = 0
    errors = 0
    print("Test: " + str(total_tests) + " tests using k = " + str(k) + " and " + str(choosen_distance) + " distance")
    for i in range(total_tests):
        # Selecao aleatoria de linha pra teste
        random_number = random.randint(0, 5893)
        query = dataset.loc[random_number].tolist()
        query.pop(len(query) - 1)
        if dataset.loc[random_number][13] == knn(dataset, query, choosen_distance, k, p):
            hits += 1
        else:
            errors += 1
    print("Hits = " + str(hits) + " - Errors: " + str(errors) + " - Average: " + str((hits / (hits + errors)) * 100) + "%")
    print()


distance_type = ["minkowski", "euclidian", "manhattan"]

test(10, 1, distance_type[0], 3)
test(10, 1, distance_type[1])
test(10, 1, distance_type[2])
test(10, 100, distance_type[0], 3)
test(10, 100, distance_type[1])
test(10, 100, distance_type[2])
test(100, 100, distance_type[0], 3)
test(100, 100, distance_type[1])
test(100, 100, distance_type[2])
