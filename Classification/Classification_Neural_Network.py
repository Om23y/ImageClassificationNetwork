import numpy as np
from Classification_dataprocess import X_train, y_train
import pandas as pd






# Hyper Parameters

learning_rate = 0.00001 # Crazy how much this changes the model
epochs = 50
batch_size = 25
momentum_constant = 0.4

Size_Of_First_Hidden_Layer = 20
Size_Of_Second_Hidden_Layer = 30
Size_Of_Third_Hidden_Layer = 20



printEpochTimes = 10 # Not related to model

Size_Of_Output_Layer = y_train.shape[1] 


input_length = len(X_train.T)


weights_0 = np.random.uniform(-1, 1, (input_length, Size_Of_First_Hidden_Layer))
weights_1 = np.random.uniform(-1, 1, (Size_Of_First_Hidden_Layer, Size_Of_Second_Hidden_Layer))
weights_2 = np.random.uniform(-1, 1, (Size_Of_Second_Hidden_Layer, Size_Of_Third_Hidden_Layer))
weights_3 = np.random.uniform(-1, 1, (Size_Of_Third_Hidden_Layer, Size_Of_Output_Layer))




def loss_function(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Initialize biases
bias_0 = np.zeros(Size_Of_First_Hidden_Layer)
bias_1 = np.zeros(Size_Of_Second_Hidden_Layer)
bias_2 = np.zeros(Size_Of_Third_Hidden_Layer)
bias_3 = np.zeros(Size_Of_Output_Layer)



def print_type(var): #To help me
    if isinstance(var, pd.DataFrame):
        print("The variable is a pandas DataFrame.")
    elif isinstance(var, np.ndarray):
        print("The variable is a numpy array.")
    else:
        print("The variable is neither a pandas DataFrame nor a numpy array. It is a", type(var))


def runNetworkForTrain(X):

    layer_0 = X
    layer_1 = np.maximum(0, np.dot(layer_0,weights_0) + bias_0)
    layer_2 = np.maximum(0, np.dot(layer_1,weights_1) + bias_1)
    layer_3 = np.maximum(0,np.dot(layer_2, weights_2) + bias_2)

        
    output_layer = softmax(np.dot(layer_3,weights_3) + bias_3)
   
    return output_layer, layer_3, layer_2, layer_1, layer_0


def runNetwork(X):

    layer_0 = X
    layer_1 = np.maximum(0, np.dot(layer_0,weights_0) + bias_0)
    layer_2 = np.maximum(0, np.dot(layer_1,weights_1) + bias_1)
    layer_3 = np.maximum(0,np.dot(layer_2, weights_2) + bias_2)

        
    output_layer = softmax(np.dot(layer_3,weights_3) + bias_3)

    return output_layer



def softmax(x):
    if x.ndim == 1:
        # Handle 1D array by converting it to 2D array with shape (1, N)
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def batchSelector(X, Y, size_of_Batch):
    X_shape = X.shape[0]
    random_batch_indexes = np.random.randint(0, X_shape-1, size=size_of_Batch)
    
    batch = None
    batch_Correct_Values = None
    for i in random_batch_indexes:
        row_batch = X.iloc[i+1]
        
        value = Y.iloc[i+1]
        if batch is None:
            batch = row_batch
            batch_Correct_Values = value
        else:
            batch = np.vstack((batch, row_batch))  # Add each batch as a new 
            batch_Correct_Values = np.vstack((batch_Correct_Values, value))
    
    return batch, batch_Correct_Values, random_batch_indexes



for epoch in range(epochs):
    X_train_Remaining = X_train
    y_train_Remaining = y_train
    old_update_weight_0 = old_update_weight_1 = old_update_weight_2 = old_update_weight_3 = old_update_weight_4 = 0
    output_layer_error_array = np.array([])
    for batches in range(round(X_train.shape[0] / batch_size)):
        
        X_Batch, y_Batch, random_batch_indexes = batchSelector(X_train_Remaining, y_train_Remaining, batch_size) # Implement Stochastic Functionality
        X_train_Remaining = X_train_Remaining.drop(random_batch_indexes, errors='ignore')
        y_train_Remaining = y_train_Remaining.drop(random_batch_indexes, errors='ignore')
        
        for i in range(X_Batch.shape[0]):
        
            input_layer = X_Batch[i]
            
            output_layer, layer_3, layer_2, layer_1, layer_0 = runNetworkForTrain(input_layer)
        
            loss = cross_entropy_loss(y_Batch[i], output_layer)
            output_error = 2*(output_layer - y_Batch[i]).T
            output_error = output_error.astype(np.float64)
         
            layer_3_error = weights_3.dot(output_error)
           
            layer_2_error = weights_2.dot(layer_3_error)
            layer_1_error = weights_1.dot(layer_2_error)
            output_layer_error_array = np.append(output_layer_error_array, loss) 
            
            update_factor_3 = np.outer(layer_3, output_error).astype(np.float64)
            weights_3 -= learning_rate * ((1 - momentum_constant) * update_factor_3 + momentum_constant * old_update_weight_3)
            old_update_weight_3 = update_factor_3
           
            bias_3 -= learning_rate * output_error.flatten()
            
            update_factor_2 = np.outer(layer_2, layer_3_error).astype(np.float64)
            
            weights_2 -= learning_rate * ((1 - momentum_constant) * update_factor_2 + momentum_constant * old_update_weight_2)
            old_update_weight_2 = update_factor_2
            bias_2 -= learning_rate * layer_3_error.flatten()

            update_factor_1 = np.outer(layer_1, layer_2_error).astype(np.float64)
            weights_1 -= learning_rate * ((1 - momentum_constant) * update_factor_1 + momentum_constant * old_update_weight_1)
            old_update_weight_1 = update_factor_1
            bias_1 -= learning_rate * layer_2_error.flatten()

            update_factor_0 = np.outer(layer_0, layer_1_error).astype(np.float64)
            weights_0 -= learning_rate * ((1 - momentum_constant) * update_factor_0 + momentum_constant * old_update_weight_0)
            old_update_weight_0 = update_factor_0
            bias_0 -= learning_rate * layer_1_error.flatten()

    
    loss = np.sum(output_layer_error_array)
    
    
  
    if (epoch+1) % int(round(epochs/printEpochTimes)) == 0 or epoch+1 == epochs:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss/output_layer_error_array.shape[0]}')
        output_layer_error_array = np.array([])


