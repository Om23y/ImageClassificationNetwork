from Classification_dataprocess import X_test, y_test;
import numpy as np;
from Classification_Neural_Network import runNetwork;
from Classification_Neural_Network import cross_entropy_loss;

def predictedClass(prediction):
    if prediction[0] > prediction[1] and prediction[0] > prediction[2]:
        return "Class 1"
    elif prediction[1] > prediction[0] and prediction[1] > prediction[2]:
        return "Class 2"
    else:
        return "Class 3"
print("\nHere in my NN for predicting the grade of students on an exam given other parameters")

output_layer_array_test = np.array([])
output_layer_array_test = np.empty((0, 3))  # Initialize an empty array with the appropriate shape

for i in range(X_test.shape[0]):
    output_layer = runNetwork(X_test.iloc[i])
    output_layer_array_test = np.vstack((output_layer_array_test, output_layer))  # Add output_layer as a row

print(f'\nTesting Cross Entropy Loss: {cross_entropy_loss(y_test, output_layer_array_test)}')


print("\nTo illustrate the network at work, let's compare a few predicted testing values to their actual values:")

print("\n\nFirst is row 2:")
print("Predicted: " + predictedClass(output_layer_array_test[2]))
print("Actual: " + y_test.columns[(y_test.iloc[2] == 1)][0])

print("\n\nFirst is row 3:")
print("Predicted: " + predictedClass(output_layer_array_test[3]))
print("Actual: " + y_test.columns[(y_test.iloc[3] == 1)][0])

print("\n\nFirst is row 4:")
print("Predicted: " + predictedClass(output_layer_array_test[4]))
print("Actual: " + y_test.columns[(y_test.iloc[4] == 1)][0])

print("\n\nFirst is row 5:")
print("Predicted: " + predictedClass(output_layer_array_test[5]))
print("Actual: " + y_test.columns[(y_test.iloc[5] == 1)][0])

print("\n\nFirst is row 7:")
print("Predicted: " + predictedClass(output_layer_array_test[7]))
print("Actual: " + y_test.columns[(y_test.iloc[7] == 1)][0])

print("\n\nFirst is row 8:")
print("Predicted: " + predictedClass(output_layer_array_test[8]))
print("Actual: " + y_test.columns[(y_test.iloc[8] == 1)][0])

print("\n\nFirst is row 9:")
print("Predicted: " + predictedClass(output_layer_array_test[9]))
print("Actual: " + y_test.columns[(y_test.iloc[9] == 1)][0])

print("\nNext is row 15:")
print("Predicted: " + predictedClass(output_layer_array_test[15]))
print("Actual: " + y_test.columns[(y_test.iloc[15] == 1)][0])

print("\nNext is row 20:")
print("Predicted: " + predictedClass(output_layer_array_test[20]))
print("Actual: " + y_test.columns[(y_test.iloc[20] == 1)][0])