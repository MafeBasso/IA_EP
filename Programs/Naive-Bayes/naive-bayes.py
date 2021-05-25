import pandas
import numpy
import math


# Leitura do dataset
def read_dataset(dataset):
    return pandas.read_csv(dataset)


def naive_bayes(dataset, query, minimal_prob = 1e-7):
    class_id = dataset.columns.size # seleciona o id da coluna das classes do dataset
    # print(class_id)

    classes = dataset[dataset.columns[class_id - 1]].unique()   # seleciona as classes únicas da coluna de classes do dataset
    # print(classes)

    probabilities = [0] * classes.size  # vetor inicializado com 1 com tamanho do número de classes
    # print(probabilities)

    for i in range(classes.size):
        # Aplicar argmax para cada classe
        # log(v1) = log(Produtório_j(P(Sim|Atrr_j) * P(Sim))
        #         = log(P(Sim|Atrr_1)) + ... + log(P(Sim|Atrr_j)) + log(P(Sim))
        # log(v2) = log(Produtório_j(P(Não|Atrr_j) * P(Não))
        #         = log(P(Não|Atrr_1)) + ... + log(P(Não|Atrr_j)) + log(P(Não))
        # exp(log(v1)) = v1
        # exp(log(v2)) = v2

        # print(dataset[dataset.columns[class_id - 1]]) # mostra a ultima coluna
        # print(classes[i]) # mostra a classe sendo avaliada
        # print(sum(dataset[dataset.columns[class_id - 1]] == classes[i])) # mostra quantas vezes a classe avaliada aparece na ultima coluna
        p_classe = sum(dataset[dataset.columns[class_id - 1]] == classes[i]) / len(dataset) # probabilidade de uma classe
        # print(p_classe)

        probabilities[i] += math.log(p_classe)
        # print(probabilities[i])

    # print(probabilities) # mostra a probabilidade de cada classe

        # print(len(query))
        for j in range(len(query)):
            attr_j = query[j]
            # print("dataset[dataset.columns[j]] = " + str(dataset[dataset.columns[j]]))
            # print("attr_j = " + str(attr_j))
            # print("dataset[dataset.columns[j]] == attr_j : " + str(dataset[dataset.columns[j]] == attr_j))
            # print("dataset[dataset.columns[class_id - 1]] : " + str(dataset[dataset.columns[class_id - 1]]))
            # print("classes[i] : " + str(classes[i]))
            if attr_j != "?": # verifica se o atributo é válido
                p_attr_j_classe = 0
                p_classes = 0
                for k in range(len(dataset)):
                    # print(dataset.loc[k][dataset.columns[j]])
                    if dataset.loc[k][dataset.columns[j]] == attr_j and dataset.loc[k][dataset.columns[class_id - 1]] == classes[i]:
                        p_attr_j_classe += 1
                    if dataset.loc[k][dataset.columns[class_id - 1]] == classes[i]:
                        p_classes += 1
                # print(str(attr_j) + " && " + str(classes[i]) + " = " + str(p_attr_j_classe))
                # print(str(classes[i]) + " = " + str(p_classes))
                if p_attr_j_classe == 0:
                    p_attr_j_classe = minimal_prob
                probabilities[i] += math.log(p_attr_j_classe / p_classes)
        # print(p_attr_j_classe)


    probabilities[0] = math.exp(probabilities[0])
    probabilities[1] = math.exp(probabilities[1])

    probabilities /= sum(numpy.array(probabilities))

    ret = list()
    ret.append(classes)
    ret.append(probabilities)
    return ret


dataset = read_dataset(dataset="..\..\Datasets\Dataset_tenis.csv")
query = ["?", "Quente", "Alta", "?"]
classes, probabilities = naive_bayes(dataset=dataset, query=query)
print(classes)
print(probabilities)