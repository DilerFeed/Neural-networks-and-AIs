import time
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt 

ITERATIONS = 100000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

Time = time.time()
err_sum, accuracy, times, iters = [], [], [], []

for iteration in range(ITERATIONS):
    
    input_layer = training_inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_outputs - outputs
    
    adjustments = error * sigmoid_derivative(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustments)
    
    if (iteration / (ITERATIONS / 100)) % 2 == 0:
        for err in error:
            err_sum.append(abs(err[0]))
        accuracy.append((1 - mean(err_sum)) * 100)
        times.append(time.time() - Time)
        iters.append(iteration)
        
        
#plt.plot(times, accuracy)
plt.plot(iters, accuracy) 

#plt.xlabel('Time (s)')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)') 

plt.show() 

print('Synaptic weights after training')
print(synaptic_weights)
    
print('Outputs after training: ')
print(outputs)