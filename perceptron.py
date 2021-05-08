import random

import numpy
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_dataset(dataset):
    return pandas.read_csv(dataset)


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
        if column == "target":
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


def minkowski_distance(query, row, columns, p):
    subtraction = query - row[0:(columns - 1)]
    internal_pow = subtraction ** p
    sumatory = sum(internal_pow)
    external_pow = sumatory ** (1 / p)
    return external_pow


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


def knn(dataset, query, distance_type, k=1, p=2):
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

def f(net, threshold = 0.5):
    return 1 if net >= threshold else 0

def f2(net, threshold = 0.1):
    return 1 if net >= threshold else -1

def perceptron(dataset, fnet, eta = 0.1, threshold = 0.001):
    columns = dataset.columns.size
    X = dataset.iloc[:,:-1]
    Y = dataset.iloc[:,columns - 1:columns]
    weights = numpy.random.uniform(-0.5, 0.5, X.columns.size + 1)
    print("Pesos iniciais: " + str(weights))

    j = 0
    sqerror = 2 * threshold
    while (sqerror > threshold):
        j += 1
        sqerror = 0
        for i in range(len(X.index)):
            input = X.loc[i]
            input["theta"] = 1
            expected = Y.loc[i]

            obtained = fnet(net = sum(weights * input))
            # print(obtained)
            # print(expected)
            error = expected - obtained
            # print(error)
            sqerror = sqerror + (error[0] ** 2)
            dE2 = 2 * error[0] * input * (-1)

            weights = weights - eta * dE2
        sqerror = sqerror / len(X.index)
    print("Pesos finais: " + str(weights.to_numpy()))
    print("Épocas: " + str(j))
    return weights

def perceptron_test_1(x, weights, fnet = f):
    x.append(1)
    return f(net = sum(weights * x))

def perceptron_test_2(x, weights, fnet = f):
    x.append(1)
    return "Escuro (-1)" if f2(net = sum(weights * x)) == -1 else "Claro (1)"

# Lê e imprime o dataset original
# print_dataset(dataset)
# print("weights = " + str(weights))

def test_1(dataset, test):
    print(test)
    weights = perceptron(dataset, f)
    print("[0,0] = " + str(perceptron_test_1([0, 0], weights)))
    print("[0,1] = " + str(perceptron_test_1([0, 1], weights)))
    print("[1,0] = " + str(perceptron_test_1([1, 0], weights)))
    print("[1,1] = " + str(perceptron_test_1([1, 1], weights)))
    print()

def test_2(dataset):
    print("Colors")
    weights = perceptron(dataset, f2)
    # print(weights)
    print("[-1, -1, -1, -1] = " + str(perceptron_test_2([-1, -1, -1, -1], weights, f2)))
    print("[-1, -1, -1, 1] = " + str(perceptron_test_2([-1, -1, -1, 1], weights, f2)))
    print("[-1, -1, 1, -1] = " + str(perceptron_test_2([-1, -1, 1, -1], weights, f2)))
    print("[-1, -1, 1, 1] = " + str(perceptron_test_2([-1, -1, 1, 1], weights, f2)))
    print("[-1, 1, -1, -1] = " + str(perceptron_test_2([-1, 1, -1, -1], weights, f2)))
    print("[-1, 1, -1, 1] = " + str(perceptron_test_2([-1, 1, -1, 1], weights, f2)))
    print("[-1, 1, 1, -1] = " + str(perceptron_test_2([-1, 1, 1, -1], weights, f2)))
    print("[-1, 1, 1, 1] = " + str(perceptron_test_2([-1, 1, 1, 1], weights, f2)))
    print("[1, -1, -1, -1] = " + str(perceptron_test_2([1, -1, -1, -1], weights, f2)))
    print("[1, -1, -1, 1] = " + str(perceptron_test_2([1, -1, -1, 1], weights, f2)))
    print("[1, -1, 1, -1] = " + str(perceptron_test_2([1, -1, 1, -1], weights, f2)))
    print("[1, -1, 1, 1] = " + str(perceptron_test_2([1, -1, 1, 1], weights, f2)))
    print("[1, 1, -1, -1] = " + str(perceptron_test_2([1, 1, -1, -1], weights, f2)))
    print("[1, 1, -1, 1] = " + str(perceptron_test_2([1, 1, -1, 1], weights, f2)))
    print("[1, 1, 1, -1] = " + str(perceptron_test_2([1, 1, 1, -1], weights, f2)))
    print("[1, 1, 1, 1] = " + str(perceptron_test_2([1, 1, 1, 1], weights, f2)))

dataset = read_dataset("Dataset_OR.csv")
test_1(dataset, "OR")

dataset = read_dataset("Dataset_AND.csv")
test_1(dataset, "AND")

dataset = read_dataset("Dataset_Colors.csv")
test_2(dataset)