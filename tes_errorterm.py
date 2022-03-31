
from utils import *
from Backpropagation import *

def main():
    # y true
    # n layer
    # output per layer
    # net per layer
    # array neuron layer
    # array activation
    # weight per layer

    y_true = [[1,0,0]]
    n_layer = 2
    output_per_layer = [[1], [0.01, 1, 0.5]]
    net_per_layer = [[[10]], [[-5,10,0]]]
    array_neuron_layer = [1,3]
    array_activation = ["sigmoid", "sigmoid"]
    weight_per_layer = [[-20, -20], [[20], [-10], [10]]]
    input_data = [[1,0]]

    backprop = Backpropagation(n_layer = n_layer, array_neuron_layer=array_neuron_layer, array_activation=array_activation, learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=1)
    backprop.weight_per_layer=weight_per_layer
    backprop.array_activation = array_activation
    backprop.n_layer = n_layer
    backprop.net_per_layer = net_per_layer
    backprop.array_neuron_layer = array_neuron_layer
    backprop.array_activation = array_activation
    backprop.output_per_layer = output_per_layer
    
    backprop.calculateErrorTerm(y_true=y_true)
    print(backprop.error_term)
    # expected data udah sesuai dengan spreadsheet di excel
    # backprop.backpropagation(input_data, y_true)

    # predicted = backprop.predict(inputData['input'])

if __name__ == "__main__":
    main()