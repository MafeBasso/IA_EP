import numpy
import pandas
import math

# # Leitura do dataset
# def read_dataset(dataset):
#     return pandas.read_csv(dataset)

# Cálculo de fnet
def fnet(net):
    return (1 / (1 + math.exp(-net)))

# Cálculo de fnet'
def d_fnet(f_net):
    return (f_net * (1 - f_net))

class Model:
    def __init__(
            self,
            input_layer,
            hidden_layer,
            output_layer,
            activation_function,
            d_activation_function,
            hidden_weights,
            output_weights
    ):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights

    def print_model(self):
        print("Model:")
        print("Input Layer = " + str(self.input_layer))
        print("Hidden Layer = " + str(self.hidden_layer))
        print("Output Layer = " + str(self.output_layer))
        print("Activation Function = " + str(self.activation_function))
        print("Derived Activation Function = " + str(self.d_activation_function))
        print("Hidden Weights = \n" + str(self.hidden_weights))
        print("Output Weights = \n" + str(self.output_weights))
        print()


def create_model(input_layer = 2, hidden_layer = 2, output_layer = 1, activation_function = fnet, d_activation_function = d_fnet):
    hidden_weights = numpy.random.uniform(-0.5, 0.5, (hidden_layer, input_layer + 1))
    output_weights = numpy.random.uniform(-0.5, 0.5, (output_layer, hidden_layer + 1))

    return Model(input_layer, hidden_layer, output_layer, activation_function, d_activation_function, hidden_weights, output_weights)


class ModelForward:
    def __init__(
            self,
            net_h_p,
            f_net_h_p,
            net_o_p,
            f_net_o_p
    ):
        self.net_h_p = net_h_p
        self.f_net_h_p = f_net_h_p
        self.net_o_p = net_o_p
        self.f_net_o_p = f_net_o_p

    def print_model(self):
        print("Model Forward:")
        print("net_h_p = \n" + str(self.net_h_p))
        print("f_net_h_p = \n" + str(self.f_net_h_p))
        print("net_o_p Layer = \n" + str(self.net_o_p))
        print("f_net_o_p = \n" + str(self.f_net_o_p))
        print()


def create_model_forward(model, p):
    p.append(1)

    print("model.hidden_weights = \n" + str(model.hidden_weights))
    print("p = \n" + str(p))
    # Hidden Layer
    temp = model.hidden_weights * p
    print("temp = \n" + str(temp))
    temp = sum(sum(temp))
    print("temp = \n" + str(temp))
    net_h_p = 0
    for i in range(len(temp)):
        net_h_p += temp[i]
    print("net_h_p = \n" + str(net_h_p))
    f_net_h_p = model.activation_function(sum(model.hidden_weights * p))
    print(f_net_h_p)

    # Output Layer
    f_net_h_p = numpy.c_[f_net_h_p, numpy.ones((len(f_net_h_p), 1))]

    net_o_p = model.output_weights * f_net_h_p
    f_net_o_p = model.output_weights * f_net_h_p
    for i in range(len(f_net_o_p)):
        for j in range(len(f_net_o_p)):
            f_net_o_p[i][j] = model.activation_function(f_net_o_p[i][j])

    return ModelForward(net_h_p, f_net_h_p, net_o_p, f_net_o_p)


model = create_model()
model.print_model()
model_forward = create_model_forward(model, [2, 2])
model_forward.print_model()

# # Cálculo de fnet para os datasets de imagem
# def fnet_images(net, threshold = 0.1):
#     return 1 if net >= threshold else -1
#
# # Treino do Perceptron
# def perceptron_training(dataset, fnet, eta = 0.1, threshold = 1e-3):
#     # columns nos retorna o numero de colunas do dataset
#     columns = dataset.columns.size
#
#     # selecionamos no training_dataset o nosso dataset excluindo a coluna de respostas e o expected values corresponde a coluna respostas
#     training_dataset = dataset.iloc[:,:-1]
#     expected_values = dataset.iloc[:,columns - 1:columns]
#
#     weights = numpy.random.uniform(-0.5, 0.5, training_dataset.columns.size + 1)
#     print("Pesos iniciais: " + str(weights))
#
#     cycles = 0
#     sqerror = 2 * threshold
#
#     # quebramos o while quando o erro for menor que o mínimo aceito
#     while (sqerror > threshold):
#         cycles += 1
#         sqerror = 0
#
#         # iteramos o nosso dataset para calcular a fnet de cada teste
#         for i in range(len(training_dataset.index)):
#             input = training_dataset.loc[i]
#
#             # correcao do tamanho do input
#             input["theta"] = 1
#
#             expected = expected_values.loc[i]
#
#             # obtemos a fnet para os valores selecionados do dataset nesta iteração
#             obtained = fnet(net = sum(weights * input))
#
#             error = expected - obtained
#             sqerror = sqerror + (error[0] ** 2)
#             dE2 = 2 * error[0] * input * (-1)
#
#             weights = weights - eta * dE2
#
#         # normalizacao quadratica do erro
#         sqerror = sqerror / len(training_dataset.index)
#     print("Pesos finais: " + str(weights.to_numpy()))
#     print("Épocas: " + str(cycles))
#     return weights
#
# def perceptron_test_logicals(input, weights, fnet = fnet_logical):
#     input.append(1)
#     return fnet_logical(net = sum(weights * input))
#
# def perceptron_test_images(input, weights, fnet = fnet_logical):
#     input.append(1)
#     return "Escuro (-1)" if fnet_images(net = sum(weights * input)) == -1 else "Claro (1)"
#
# def test_logicals(dataset, test):
#     print(test)
#     weights = perceptron_training(dataset, fnet_logical)
#     print("[0,0] = " + str(perceptron_test_logicals([0, 0], weights)))
#     print("[0,1] = " + str(perceptron_test_logicals([0, 1], weights)))
#     print("[1,0] = " + str(perceptron_test_logicals([1, 0], weights)))
#     print("[1,1] = " + str(perceptron_test_logicals([1, 1], weights)))
#     print()
#
# def test_images(dataset):
#     print("Colors")
#     weights = perceptron_training(dataset, fnet_images)
#     print("[-1, -1, -1, -1] = " + str(perceptron_test_images([-1, -1, -1, -1], weights, fnet_images)))
#     print("[-1, -1, -1, 1] = " + str(perceptron_test_images([-1, -1, -1, 1], weights, fnet_images)))
#     print("[-1, -1, 1, -1] = " + str(perceptron_test_images([-1, -1, 1, -1], weights, fnet_images)))
#     print("[-1, -1, 1, 1] = " + str(perceptron_test_images([-1, -1, 1, 1], weights, fnet_images)))
#     print("[-1, 1, -1, -1] = " + str(perceptron_test_images([-1, 1, -1, -1], weights, fnet_images)))
#     print("[-1, 1, -1, 1] = " + str(perceptron_test_images([-1, 1, -1, 1], weights, fnet_images)))
#     print("[-1, 1, 1, -1] = " + str(perceptron_test_images([-1, 1, 1, -1], weights, fnet_images)))
#     print("[-1, 1, 1, 1] = " + str(perceptron_test_images([-1, 1, 1, 1], weights, fnet_images)))
#     print("[1, -1, -1, -1] = " + str(perceptron_test_images([1, -1, -1, -1], weights, fnet_images)))
#     print("[1, -1, -1, 1] = " + str(perceptron_test_images([1, -1, -1, 1], weights, fnet_images)))
#     print("[1, -1, 1, -1] = " + str(perceptron_test_images([1, -1, 1, -1], weights, fnet_images)))
#     print("[1, -1, 1, 1] = " + str(perceptron_test_images([1, -1, 1, 1], weights, fnet_images)))
#     print("[1, 1, -1, -1] = " + str(perceptron_test_images([1, 1, -1, -1], weights, fnet_images)))
#     print("[1, 1, -1, 1] = " + str(perceptron_test_images([1, 1, -1, 1], weights, fnet_images)))
#     print("[1, 1, 1, -1] = " + str(perceptron_test_images([1, 1, 1, -1], weights, fnet_images)))
#     print("[1, 1, 1, 1] = " + str(perceptron_test_images([1, 1, 1, 1], weights, fnet_images)))
#
# dataset = read_dataset("Dataset_OR.csv")
# test_logicals(dataset, "OR")
#
# dataset = read_dataset("Dataset_AND.csv")
# test_logicals(dataset, "AND")
#
# dataset = read_dataset("Dataset_Colors.csv")
# test_images(dataset)