from utils import *

def main():
    fileModel = input("Masukan file model : ")
    fileInput = input("Masukan file input : ")

    modelData = readFile(fileModel)
    inputData = readFile(fileInput)

    # Print model here
    printModel(modelData)
    
    output = predictFeedForward(modelData, inputData)
    
    print("Matrix hasil prediksi : ")
    print(output)
    promptExport = input("Do you want to export output data ? (y/n) ")
    if (promptExport.lower() == "y"):
        filename = input("Input filename : (Not with extension) ")
        exportOutput(output, filename)
        return
        

if __name__ == "__main__":
    main()
