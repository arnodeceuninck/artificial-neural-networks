import torch
from sklearn.datasets import make_blobs
from matplotlib import cm, pyplot as plt

import numpy as np

# This function plots generated randon synthetic dataset
# This part of code related to visualize the dataset has been obtained from Matplotlib library tutorial
def plot_dataset(x,y):
    plt.figure()
    y_unique = np.unique(y)
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = x[y == this_y]
        plt.scatter(this_X[:, 0],this_X[:, 1],s=50,c=color[np.newaxis, :],alpha=0.5,edgecolor="k",label="Class %s" % this_y)
    plt.legend(loc="best")
    plt.title("Data")
    plt.show()


dtype = torch.double
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is the batch size, D_in is input dimension
# H is hidden dimension, D_out is output dimension
N, D_in, H, D_out = 500, 2, 1000, 3

# generating random input and output (label) data
centers = [(-5, -5), (-5, -3), (-3, -3)]
x, y = make_blobs(n_samples=N, centers=centers, n_features=D_in,random_state=0)
x[:,0] = (x[:,0] - np.min(x[:,0])) / (np.max(x[:,0]) - np.min(x[:,0]))
x[:,1] = (x[:,1] - np.min(x[:,1])) / (np.max(x[:,1]) - np.min(x[:,1]))
# plot data to see the distribution of data
plot_dataset(x,y)

# Transfer to pytorch. since we are going to use pytorch package instead of numpy
x = torch.from_numpy(x)
# y labels are discrete numbers, we are going to map them in a categorical representation
y_cat = np.zeros((y.shape[0],D_out))
for item_index, class_index in enumerate(y):
    y_cat[item_index,class_index] = 1.0
y = torch.from_numpy(y_cat)


# Create weights (parameters) randomly
random_seed = 2 # changing this changes the initial weights, which makes your final loss different
torch.manual_seed(random_seed)
w1 = torch.randn(D_in, H, device = device, dtype=dtype)
w2 = torch.randn(H, D_out, device = device, dtype=dtype)

lr = 1e-10

loss_list = []
for itr in range(500):

    # Problem 1 - forward pass: compute prediction
    z1 = x.mm(w1) # Matrix multiplication between x and w1
    z1_act = z1.clamp(min=0) # Reulu activation function
    output = z1_act.mm(w2) # Multiply with second weights
    # end of forward pass



    # Problem 1 - compute and print loss value each 10 steps
    probability = torch.nn.Softmax(dim=1)(output)
    l = (output-y).pow(2).sum().item()
    loss_list.append(l)
    # end of loss computation

    if itr % 10 == 0:
        # print loss function
        pass # remove this line after completing if block

    # Problem 1- Backprop to compute gradients of w1 and w2 w.r.t loss value
        # gradient of loss value w.r.t w
        # 1- gradient of loss function w.r.t prediction


        # 2- gradient of prediction w.r.t w2
    grad_output = 2*(output-y)
    grad_w2 = z1_act.t().mm(grad_output)

    # gradient of loss value w.r.t w1
    grad_o1_act = grad_output.mm(w2.t())
    grad_o1_act[z1_act<0] = 0
    # grad_o1_act[z1_act>0] = 1
    grad_w1 = x.t().mm(grad_o1_act)

        # 1- gradient of loss function w.r.t prediction

        # 2- gradient of output w.r.t output of hidden layer

        # 3- gradient of output of hidden layer w.r.t ReLu function

        # 4- gradient of previous step (i.e. result from gradient of ReLu function) w.r.t w1


    # Problem 2- update weights (parameters) using gradient descent
        # 1- update w1
    w1 += -lr*grad_w1

        # 2- update w2
    w2 += -lr*grad_w2

# plot loss values over all iteration
print('loss values per iteration')
print(loss_list)
plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()
plt.cla()