import numpy
import pandas

# Leitura do dataset
def read_dataset(dataset):
    return pandas.read_csv(dataset)

# Cálculo de fnet para os dataset lógicos
def fnet_logical(net, threshold = 0.5):
    return 1 if net >= threshold else 0

# Cálculo de fnet para os datasets de imagem
def fnet_images(net, threshold = 0.1):
    return 1 if net >= threshold else -1

# Treino do Perceptron
def perceptron_training(dataset, fnet, eta = 0.1, threshold = 1e-3):
    # columns nos retorna o numero de colunas do dataset
    columns = dataset.columns.size

    # selecionamos no training_dataset o nosso dataset excluindo a coluna de respostas e o expected values corresponde a coluna respostas
    training_dataset = dataset.iloc[:,:-1]
    expected_values = dataset.iloc[:,columns - 1:columns]

    weights = numpy.random.uniform(-0.5, 0.5, training_dataset.columns.size + 1)
    print("Pesos iniciais: " + str(weights))

    cycles = 0
    sqerror = 2 * threshold

    # quebramos o while quando o erro for menor que o mínimo aceito
    while (sqerror > threshold):
        cycles += 1
        sqerror = 0

        # iteramos o nosso dataset para calcular a fnet de cada teste
        for i in range(len(training_dataset.index)):
            input = training_dataset.loc[i]

            # correcao do tamanho do input
            input["theta"] = 1

            expected = expected_values.loc[i]

            # obtemos a fnet para os valores selecionados do dataset nesta iteração
            obtained = fnet(net = sum(weights * input))

            error = expected - obtained
            sqerror = sqerror + (error[0] ** 2)
            dE2 = 2 * error[0] * input * (-1)

            weights = weights - eta * dE2

        # normalizacao quadratica do erro
        sqerror = sqerror / len(training_dataset.index)
    print("Pesos finais: " + str(weights.to_numpy()))
    print("Épocas: " + str(cycles))
    return weights

def perceptron_test_logicals(input, weights, fnet = fnet_logical):
    input.append(1)
    return fnet_logical(net = sum(weights * input))

def perceptron_test_images(input, weights, fnet = fnet_logical):
    input.append(1)
    return "Escuro (-1)" if fnet_images(net = sum(weights * input)) == -1 else "Claro (1)"

def test_logicals(dataset, test):
    print(test)
    weights = perceptron_training(dataset, fnet_logical)
    print("[0,0] = " + str(perceptron_test_logicals([0, 0], weights)))
    print("[0,1] = " + str(perceptron_test_logicals([0, 1], weights)))
    print("[1,0] = " + str(perceptron_test_logicals([1, 0], weights)))
    print("[1,1] = " + str(perceptron_test_logicals([1, 1], weights)))
    print()

def test_images(dataset):
    print("Colors")
    weights = perceptron_training(dataset, fnet_images)
    print("[-1, -1, -1, -1] = " + str(perceptron_test_images([-1, -1, -1, -1], weights, fnet_images)))
    print("[-1, -1, -1, 1] = " + str(perceptron_test_images([-1, -1, -1, 1], weights, fnet_images)))
    print("[-1, -1, 1, -1] = " + str(perceptron_test_images([-1, -1, 1, -1], weights, fnet_images)))
    print("[-1, -1, 1, 1] = " + str(perceptron_test_images([-1, -1, 1, 1], weights, fnet_images)))
    print("[-1, 1, -1, -1] = " + str(perceptron_test_images([-1, 1, -1, -1], weights, fnet_images)))
    print("[-1, 1, -1, 1] = " + str(perceptron_test_images([-1, 1, -1, 1], weights, fnet_images)))
    print("[-1, 1, 1, -1] = " + str(perceptron_test_images([-1, 1, 1, -1], weights, fnet_images)))
    print("[-1, 1, 1, 1] = " + str(perceptron_test_images([-1, 1, 1, 1], weights, fnet_images)))
    print("[1, -1, -1, -1] = " + str(perceptron_test_images([1, -1, -1, -1], weights, fnet_images)))
    print("[1, -1, -1, 1] = " + str(perceptron_test_images([1, -1, -1, 1], weights, fnet_images)))
    print("[1, -1, 1, -1] = " + str(perceptron_test_images([1, -1, 1, -1], weights, fnet_images)))
    print("[1, -1, 1, 1] = " + str(perceptron_test_images([1, -1, 1, 1], weights, fnet_images)))
    print("[1, 1, -1, -1] = " + str(perceptron_test_images([1, 1, -1, -1], weights, fnet_images)))
    print("[1, 1, -1, 1] = " + str(perceptron_test_images([1, 1, -1, 1], weights, fnet_images)))
    print("[1, 1, 1, -1] = " + str(perceptron_test_images([1, 1, 1, -1], weights, fnet_images)))
    print("[1, 1, 1, 1] = " + str(perceptron_test_images([1, 1, 1, 1], weights, fnet_images)))

dataset = read_dataset("Dataset_OR.csv")
test_logicals(dataset, "OR")

dataset = read_dataset("Dataset_AND.csv")
test_logicals(dataset, "AND")

dataset = read_dataset("Dataset_Colors.csv")
test_images(dataset)