import numpy as np
import scipy.special

class neuralNetwork():
    # iinicializar la red neuronal
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # establecer el número de nodos en cada capa de entrada, oculta y de salida
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # vincular matrices de peso, wih y who
        # los pesos dentro de las matrices son w_i_j, donde el enlace es del nodo i al nodo j en la siguiente capa
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # tasa de aprendizaje
        self.lr = learningrate

        # la función de activación es la función sigmoidea
        self.activation_function = lambda x: scipy.special.expit(x)

    # entrenar la red neuronal
    def train(self, inputs_list, targets_list):
        # convierte la lista de entradas en un arreglo 2d
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calcular señales en capa oculta
        hidden_inputs = np.dot(self.wih, inputs)
        # calcular las señales que emergen de la capa oculta
        hidden_outputs = self.activation_function(hidden_inputs)

        # calcular señales de la capa de salida final
        final_inputs = np.dot(self.who, hidden_outputs)
        # calcular las señales que emergen de la capa de salida final
        final_outputs = self.activation_function(final_inputs)
        # el error de la capa de salida es el (target - actual)
        output_errors = targets - final_outputs
        # error de capa oculta es el output_errors, dividida por pesos, recombinada en nodos ocultos
        hidden_errors = np.dot(self.who.T, output_errors)

        # actualizar los pesos de los enlaces entre la capa oculta y la de salida
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # actualizar los pesos de los enlaces entre la entrada y la capa oculta
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # consultar la red neuronal
    def query(self, inputs_list):
        # convierte la lista de entradas en un arreglo 2d
        inputs = np.array(inputs_list, ndmin=2).T

        # calcular señales en capa oculta
        hidden_inputs = np.dot(self.wih, inputs)
        # calcular las señales que emergen de la capa oculta
        hidden_outputs = self.activation_function(hidden_inputs)

        # calcular señales en la capa de salida final
        final_inputs = np.dot(self.who, hidden_outputs)
        # calcular las señales que emergen de la capa de salida final
        fianl_outputs = self.activation_function(final_inputs)

        return fianl_outputs