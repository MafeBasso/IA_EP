import numpy
import pandas
import math
import pickle

# Leitura do dataset
def read_dataset(dataset):
    dataset = pandas.read_csv(dataset)
    dataset = dataset.drop([
        "accelerometer#min",
        "accelerometer#max",
        "accelerometer#std",
        "gyroscope#min",
        "gyroscope#max",
        "gyroscope#std",
        "sound#min",
        "sound#max",
        "sound#std"],
        axis=1)
    return dataset


def pre_processing(dataset):
    # Como apenas a coluna target possui strings, ela recebe um tratamento diferente
    for column in dataset.columns:
        if column == "still" or column == "car" or column == "train" or column == "bus" or column == "walking":
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


def normalize(dataset, classes):
    # Removemos a coluna target para fazer a normalização porque ela possui apenas strings
    # dataset_values = dataset[:len(dataset) - classes]
    dataset_attributes = dataset.drop(["still", "car", "train", "bus", "walking"], axis=1)
    dataset_classes = dataset.drop([
        "time",
        "accelerometer#mean",
        "gyroscope#mean",
        "sound#mean"
        ],
        axis=1)
    normalized = ((dataset_attributes - dataset_attributes.min()) / (dataset_attributes.max() - dataset_attributes.min()))
    normalized = numpy.array(normalized)
    dataset_classes = numpy.array(dataset_classes)
    dataset = numpy.append(normalized, dataset_classes, axis=1)
    dataset = pandas.DataFrame(dataset, columns=[
        "time",
        "accelerometer#mean",
        "gyroscope#mean",
        "sound#mean",
        "still",
        "car",
        "train",
        "bus",
        "walking"
    ])
    return dataset


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
            output_weights,
            cycles = 0
    ):
        self.cycles = cycles
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


def create_model(input_layer, hidden_layer, output_layer, activation_function = fnet, d_activation_function = d_fnet):
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

    def get_result(self):
        selected = self.f_net_o_p.index(max(self.f_net_o_p))
        print(selected)
        classes = ["still", "car", "train", "bus", "walking"]
        return classes[selected]


def test_model(model, p):
    p.append(1)

    # net Hidden Layer
    net_h_p = [0] * model.hidden_layer
    for i in range(len(net_h_p)):
        net_h_p[i] = sum(model.hidden_weights[i] * p)

    # f_net Hidden Layer
    f_net_h_p = [0] * len(net_h_p)
    for i in range(len(net_h_p)):
        f_net_h_p[i] = model.activation_function(net_h_p[i])

    f_net_h_p.append(1)

    # net Output Layer
    net_o_p = [0] * model.output_layer
    for i in range(len(net_o_p)):
        net_o_p[i] = sum(model.output_weights[i] * f_net_h_p)

    # f_net Output Layer
    f_net_o_p = [0] * len(net_o_p)
    for i in range(len(f_net_o_p)):
        f_net_o_p[i] = model.activation_function(net_o_p[i])

    return ModelForward(net_h_p, f_net_h_p, net_o_p, f_net_o_p)


def backpropagation(model, dataset, eta = 0.1, threshold = 0.65):
    print("Initial Hidden Weights = ")
    print(model.hidden_weights)
    print("Initial Output Weights = ")
    print(model.output_weights)
    print("MLP is running. Please wait...")

    squared_error = 2 * threshold
    cycles = 0

    while (squared_error > threshold):
        squared_error = 0

        for i in range(len(dataset)):
            x_p = dataset.to_numpy()[i][:model.input_layer].tolist() # lista dos atributos
            y_p = dataset.to_numpy()[i][model.input_layer:dataset.columns.size] # vetor das classes

            tested_model = test_model(model, x_p) # model forwarded com as nets
            obtained_value_p = tested_model.f_net_o_p

            # Error calculation
            error = y_p - obtained_value_p
            squared_error += sum(error ** 2)

            # Training output layer
            delta_o_p = [0] * len(tested_model.f_net_o_p)
            for j in range(len(delta_o_p)):
                delta_o_p[j] = error[j] * model.d_activation_function(tested_model.f_net_o_p[j])

            # Training hidden layer
            w_o_k = [0] * len(tested_model.f_net_o_p) # matriz de listas dos pesos da output
            delta_h_p = [0] * len(tested_model.f_net_o_p) # lista de numpy array
            for j in range(len(delta_h_p)):
                w_o_k[j] = model.output_weights[j][:model.hidden_layer].tolist()
                delta_h_p[j] = delta_o_p[j] * numpy.array(w_o_k[j]) * model.d_activation_function(tested_model.f_net_h_p[j])

            # Training
            for j in range(len(delta_o_p)):
                model.output_weights[j] += eta * (delta_o_p[j] * numpy.array(tested_model.f_net_h_p)) # matriz com pesos da output

            for j in range(len(delta_h_p)):
                model.hidden_weights[j] += eta * numpy.multiply(numpy.array(delta_h_p), x_p)[j]

        squared_error /= len(dataset)
        cycles += 1
        # if (cycles % 10 == 0):
        #     print(squared_error)

    print("MLP has finished running.")
    print("Final Hidden Weights = ")
    print(model.hidden_weights)
    print("Final Output Weights = ")
    print(model.output_weights)
    print("Cycles = " + str(cycles))

    model.cycles = cycles
    save_trained_model(model)
    return model


def save_trained_model(model):
    with open("model.dat", "wb") as model_file:
        pickle.dump(model, model_file)


def create(input_layer, hidden_layer, output_layer, activation_function = fnet, d_activation_function = d_fnet):
    model = create_model(input_layer, hidden_layer, output_layer, activation_function, d_activation_function)
    return backpropagation(model, dataset)


def load_trained_model(input_layer, hidden_layer, output_layer, recreate_model = False, activation_function = fnet, d_activation_function = d_fnet):
    if recreate_model:
        return create(input_layer, hidden_layer, output_layer, activation_function, d_activation_function)
    else:
        try:
            with open("model.dat", "rb") as model_file:
                return pickle.load(model_file)
        except IOError:
            return create(input_layer, hidden_layer, output_layer, activation_function, d_activation_function)
        finally:
            model_file.close()


def get_training_and_test(dataset, percentage, lines_removed = 0):
    if lines_removed != 0:
        dataset = dataset[:len(dataset) - lines_removed]

    x = math.floor(len(dataset) * percentage)
    return dataset[:x], dataset[x:len(dataset)]


def get_result(fnets, tests):
    tests = tests.tolist()
    if fnets.index(max(fnets)) == tests.index(max(tests)):
        return True
    return False


def calculate_accuracy(model, test_dataset, classes):
    hits = 0
    errors = 0
    for i in range(test_dataset.first_valid_index(), test_dataset.last_valid_index() + 1):
        test_tuple = numpy.array(test_dataset.loc[i])
        result = test_model(model, test_tuple[:test_dataset.first_valid_index() + len(test_dataset) - classes].tolist())
        if get_result(result.f_net_o_p, test_tuple[classes - 1:]):
            hits += 1
        else:
            errors += 1
    accuracy = hits / (hits + errors) * 100
    print("Accuracy: " + str(accuracy) + "%")


# Lê e imprime o dataset original
dataset = read_dataset(dataset = "Dataset_TMD_MLP.csv")

# Faz o pré-processamento dos dados
dataset = pre_processing(dataset)

# Normaliza o dataset usando a re-escala linear
dataset = normalize(dataset, classes = 5)

# Obtém o dataset de treino e o dataset de teste com base na porcentagem escolhida
training_dataset, test_dataset = get_training_and_test(dataset = dataset, percentage = 0.7, lines_removed = 5884)

# Carrega o modelo já treinado, ou treina um novo caso ainda não exista um modelo
model = load_trained_model(input_layer = 4, hidden_layer = 5, output_layer = 5, recreate_model = False)

# Valida o dataset de testes
calculate_accuracy(model, test_dataset, classes = 5)
