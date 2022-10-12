from Neuronal_Network import neuralNetwork
import numpy as np

# número de nodos de entrada, ocultos y de salida
input_nodes = 784  # 28*28
hidden_nodes = 100
output_nodes = 10

# la tasa de aprendizaje es 0.3
learning_rate = 0.3

# crear instancia de red neuronal
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

save_wih_file = "save_model/wih.npy"
save_who_file = "save_model/who.npy"
n.wih = np.load(save_wih_file)
n.who = np.load(save_who_file)

# cargue el archivo csv de datos de prueba mnist en una lista
test_data_file = open("mnist_dataset/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# probar la red neuronal
# cuadro de mando de qué tan bien se desempeña la red, inicialmente vacío
scorecard = []
# revisar todos los registros en el conjunto de datos de prueba
for record in test_data_list:
    # dividir el registro por ',' comas
    all_values = record.split(',')
    # la respuesta correcta es el primer valor
    correct_label = int(all_values[0])
    # print(correct_label, "correct_label")
    # escalar y desplazar las entradas
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # consultar la red
    outputs = n.query(inputs)
    # el índice de mayor valor corresponde a la etiqueta
    label = np.argmax(outputs)
    # print(label, "network's answer")
    # agregar correcto o incorrecto a la lista
    if label == correct_label:
        # la respuesta de la red coincide con la respuesta correcta, agregue 1 al cuadro de mando
        scorecard.append(1)
    else:
        # la respuesta de la red no coincide con la respuesta correcta, agregue 1 al cuadro de mando
        scorecard.append(0)

# calcular la puntuación de rendimiento, la fracción de respuestas correctas
scorecard_array = np.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print("La precisión es = ", performance)

save_accuracy = "accuracy.txt"
with open(save_accuracy, 'a') as f:
    f.write("La tasa de aprendizaje es : " + str(n.lr))
    f.write("       La precisión es : " + str(performance) + '\n')