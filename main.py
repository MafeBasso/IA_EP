import pandas

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