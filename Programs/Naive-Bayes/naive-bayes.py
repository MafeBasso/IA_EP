import pandas
import numpy
import math


# Leitura do dataset
def read_dataset(dataset):
    return pandas.read_csv(dataset)


# Algoritmo Naive Bayes
def naive_bayes(dataset, query, minimal_prob=1e-7):
    class_column = dataset.columns.size  # seleciona a coluna das classes do dataset
    classes = dataset[dataset.columns[class_column - 1]].unique()  # seleciona as classes únicas da coluna de classes do dataset
    probabilities = [0] * classes.size  # inicia com 0 um vetor de probabilidades com tamanho classes.size

    for i in range(classes.size):
        class_probability = sum(dataset[dataset.columns[class_column - 1]] == classes[i]) / len(dataset)  # probabilidade da classe iterada
        probabilities[i] += math.log(class_probability)  # aplica o log na probabilidade da classe iterada para evitar convergência à zero em casos com muitas classes

        for j in range(len(query)):
            if query[j] != "?":  # verifica se o atributo atual foi passado como parâmetro
                probability_query_class = 0
                probability_class = 0

                for k in range(len(dataset)):
                    if dataset.loc[k][dataset.columns[j]] == query[j] and dataset.loc[k][dataset.columns[class_column - 1]] == classes[i]:
                        probability_query_class += 1  # conta quantas vezes o atributo atual aparece e se a classe dele é igual à classe atual
                    if dataset.loc[k][dataset.columns[class_column - 1]] == classes[i]:
                        probability_class += 1  # conta quantas vezes a classe atual aparece para oa tributo atual

                if probability_query_class == 0:  # define uma probabilidade mínima para os casos em que a probabilidade for zero
                    probability_query_class = minimal_prob

                probabilities[i] += math.log(probability_query_class / probability_class)

    # desfaz o log aplicando a exponencial e restaurando as probabilidades corretas
    probabilities[0] = math.exp(probabilities[0])
    probabilities[1] = math.exp(probabilities[1])

    probabilities /= sum(numpy.array(probabilities))  # normaliza as probabilidades

    show_results(classes, probabilities)


def show_results(classes, probabilities):
    print("Classes:")
    for i in range(len(classes)):
        print(str(classes[i]) + " = " + str(probabilities[i] * 100) + "%")


dataset = read_dataset(dataset="Dataset_tenis.csv")

query = ["?", "Quente", "Alta", "?"]

naive_bayes(dataset=dataset, query=query)
