from Backpropagation import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

def main():
    # load iris data
    inputData = load_iris()
    encoder = OneHotEncoder(sparse=False)
    reshape = inputData["target"].reshape(len(inputData["target"]), 1)
    target = encoder.fit_transform(reshape)

    bp = Backpropagation(n_layer = 2, array_neuron_layer=[2,2], array_activation=["sigmoid", "sigmoid"], learning_rate=0.01, error_threshold=1, max_iter=300, batch_size=30)

    bp.initWeightBiasRandom(inputData)

    bp.printModel()

if __name__ == "__main__":
    main()