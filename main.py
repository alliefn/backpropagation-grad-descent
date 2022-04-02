from turtle import back
from utils import *
from Backpropagation import *
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

def main():
   # load iris data
    inputData = load_iris()
    target_unencoded = inputData.target
    encoder = OneHotEncoder(sparse=False)
    reshape = inputData["target"].reshape(len(inputData["target"]), 1)
    target = encoder.fit_transform(reshape)

    # Define parameters
    n_layer = 5
    array_neuron_layer = [16,8,4, 3,3]
    array_activation = ["linear", "relu", 'linear', "relu", "sigmoid"]
    learning_rate = 0.001
    error_threshold = 0.01
    max_iter = 500
    batch_size = 1

    # create model
    backprop = Backpropagation(n_layer = n_layer, array_neuron_layer=array_neuron_layer, array_activation=array_activation, learning_rate=learning_rate, error_threshold=error_threshold, max_iter=max_iter, batch_size=batch_size)
    
    # train model
    inputData = inputData["data"].tolist()
    target = target.tolist()


    inputData, target, target_unencoded = shuffle(inputData, target, target_unencoded)
    print(inputData)
    print(target)
    backprop.backpropagation(inputData, target)
    
    #print info
    print("Info")
    backprop.printModel()
    print("-------------------------")

    #print result
    predicted = backprop.predict(inputData)
    print("Predicted Value")
    print(predicted)
    print("Real Value")
    print(target_unencoded)

    # print score accuracy
    print("Score Accuracy")
    print(score_accuracy(predicted, target_unencoded))
    conf_matrix = confusion_matrix(predicted, target_unencoded)
    print("Confusion Matrix")
    print(conf_matrix)
    print(confusion_matrix_statistics(conf_matrix))
    print()
        

if __name__ == "__main__":
    main()