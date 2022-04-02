from utils import *
from Backpropagation import *

def main():
    X_test = [[0,0], [0,1], [1,0], [1,1]]

    backprop = Backpropagation(n_layer = 2, array_neuron_layer=[2,1], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=4)
    backprop.weight_bias_layer = [ [[-10,20,20], [30,-20,-20]], [[-30,20,20]]  ]
    predicted = backprop.predictFeedForward(X_test)
    print(predicted)
    # backprop.calculateErrorTerm(y_true)

if __name__ == "__main__":
    main()