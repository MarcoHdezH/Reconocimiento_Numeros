from Neuronal_Network import neuralNetwork
import numpy as np


# número de nodos de entrada, ocultos y de salida
input_nodes = 784  # 28*28
hidden_nodes = 700
output_nodes = 10

# la tasa de aprendizaje es 0.3
learning_rate = 0.1

# crear instancia de red neuronal
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train6k.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# entrenar la red neuronal

# epochs es el número de veces que se utiliza el conjunto de datos de entrenamiento para el entrenamiento
epochs = 5
i=0

for e in range(epochs):
    # revisar todos los registros en el conjunto de datos de entrenamiento
    for record in training_data_list:
        # dividir el registro por ',' comas
        all_values = record.split(',')
        # escalar y desplazar las entradas
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # crear los valores de salida de destino (todo 0.01, experto en la etiqueta deseada que es 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] es la etiqueta de destino para este registro
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        print(i)
        i=1+i

save_wih_file = "save_model/wih.npy"
np.save(save_wih_file, n.wih)
save_who_file = "save_model/who.npy"
np.save(save_who_file, n.who)