# Problem 1 + 2
The designed architecture has 1000 input neurons and 10 output neuron, with a hidden layer of 100 neurons.

The forward pass operation is defined as follows:

$$o1 = X*W1$$ (X contains N inputs)
$$z1 = ReLu(o1)$$
$$o2 = z1*W2$$

$$L = \Sigma_{i=1}^{n} (O2_i - Y_i)^2$$

dL/dW2 = dL/dO2 * dO2/dW2

dL/dO2 = 2*(O2 - Y)
dO2/dW2 = z1

dL/dW2 = 2*(O2 - Y) * z1

dL/dW1 = dL/dO2 * dO2/dz1 * dz1/dO1 * dO1/dW1

dL/dO2 = 2*(O2 - Y)
dO2/dz1 = W2
dz1/dO1 = 1 if O1 > 0 else 0
dO1/dW1 = X

dL/dW1 = 2*(O2 - Y) * W2 * 1 if O1 > 0 else 0 * X

w1 = w1 - lr * dL/dW1
w2 = w2 - lr * dL/dW2

Changing the value of the seed, changes the initial weights, and thus also all the losses. 

## Problem 3
a) What does sign and magnitude of the gradient represent in the learning problem?

The sign of the gradient represents the direction of the steepest ascent, and the magnitude represents the steepness of the ascent.

Gradient: calculates the effect of each of the weight in the output. We want to mininmize the loss. The minimum and maximum is in the points where the gradient is 0. 

