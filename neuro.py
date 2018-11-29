import numpy as np

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
X = np.array([[0,1], [0,1], [1,0], [1,0]])     
y = np.array([[0,0,1,1]]).T

np.random.seed(1)
synapse_0 = 2*np.random.random((2,1)) - 1
print(synapse_0)

for iter in range(10000):
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))

    layer_1_error = layer_1 - y

    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0 = synapse_0 - np.dot(layer_0.T,layer_1_delta)

print("Вывод после тренировки:")
print(layer_1)