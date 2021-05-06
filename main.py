import random
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_test_list(dataset, size):
    global list_last_position
    list_last_position = len(dataset) - 1
    global list_first_position
    list_first_position = list_last_position - test_list_size + 1
    test_list = dataset.drop(dataset[dataset.index < len(dataset) - size].index)
    return test_list


def test(test_list, total_tests, k, choosen_distance, p=0):
    if total_tests > len(test_list):
        total_tests = len(test_list)
        print("total_tests > test_list.size -> total_tests = test_list.size")
    hits = 0
    errors = 0
    print("Test: " + str(total_tests) + " tests using k = " + str(k) + " and " + str(choosen_distance) + " distance")
    random_number = random.sample(test_list.index.tolist(), total_tests)
    for i in range(total_tests):
        # Selecao aleatoria de linha pra teste
        query = test_list.loc[random_number[i]].tolist()
        query.pop(len(query) - 1)
        if test_list.loc[random_number[i]][13] == knn(dataset, query, choosen_distance, k, p):
            hits += 1
        else:
            errors += 1
    print("Hits = " + str(hits) + " - Errors: " + str(errors) + " - Average: " + str(
        (hits / (hits + errors)) * 100) + "%")
    print()


# Lê e imprime o dataset original
dataset = read_dataset()

# Faz o pré-processamento dos dados
dataset = pre_processing()

# Normaliza o dataset usando a re-escala linear e imprime ele
dataset = normalize()

# Mostra a análise exploratória dos dados
# exploratory_analysis()
correlation_matrix()

# Tipos de distancias implementadas e tamanho da lista de testes (obs: a lista de testes é removida do fim do dataset)
distance_type = ["minkowski", "euclidian", "manhattan"]
test_list_size = 100

# Cria a lista de testes a partir do dataset, removendo test_list_size elementos do fim do dataset
testList = get_test_list(dataset, test_list_size)
dataset = dataset.drop(dataset[dataset.index > len(dataset) - test_list_size].index)

# Cria alguns testes alternando o número de testes, k e tipo de distância, avaliando a precisão do algoritmo
# Assinatura da função de testes (Para usar a distância de minkowski, é necessário informar p):
#   def test(testList, total_tests, k, choosen_distance, p = 0):

test(testList, 1001, 1, distance_type[0], 3)
test(testList, 10, 1, distance_type[1])
test(testList, 10, 1, distance_type[2])
test(testList, 10, 100, distance_type[0], 3)
test(testList, 10, 100, distance_type[1])
test(testList, 10, 100, distance_type[2])
test(testList, 100, 100, distance_type[0], 3)
test(testList, 100, 100, distance_type[1])
test(testList, 100, 100, distance_type[2])
