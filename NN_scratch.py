"""Making neural networks from scratch"""
# Neural network for 2 dimensional data (Regression)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
sns.set_theme()
np.random.seed(42)


# Sigmoid function
def sigmoid(weighted_sum, der=False):
    if der:
        activated_weighted_sum = weighted_sum
        return activated_weighted_sum * (1 - activated_weighted_sum)
    return 1 / (1 + np.exp(-weighted_sum))


# Hyperbolic function
def hyp(weighted_sum, der=False):
    if der:
        activated_weighted_sum = weighted_sum
        return 1 - hyp(activated_weighted_sum)**2
    return (np.exp(weighted_sum) - np.exp(-weighted_sum)) / (np.exp(weighted_sum) + np.exp(-weighted_sum))


# Loss derivative
def loss_derivative(output, activated_weighted_sum):
    return -2 * (output - activated_weighted_sum)


# Plot
def plot(x, y, pred, marker, color, save=False):
    markers = ['o', '^']
    colours = ['k', 'r']
    sns.scatterplot(x=x[:, 0], y=y[:, 0], alpha=1,
                    **{'marker': markers[marker], 'color': colours[color], 'label': 'Target'})
    plt.plot(X, pred, 'k', alpha=0.7, label="Regression line")
    plt.xlabel('Input')
    plt.ylabel('Sine function')
    plt.legend()
    if save:
        plt.savefig('Plot/{}.png'.format(epochs))
    if epochs == n:
        plt.show()
    plt.close()


# Sine function dataset
df = DataFrame(read_csv("Data.txt", delim_whitespace=True)).iloc[:, :]

# Input
X = np.array(df.iloc[:, [0]])
# Normalized input in range [0,1]
X = (X.copy() - np.min(X.copy()))/(np.max(X.copy()) - np.min(X.copy()))

# Output
Y = np.array(df.iloc[:, [1]])

# Weight Initialization
W = (0.2 * np.random.random((1, 1))) - 1
print("X", X.shape, "\nY", Y.shape, "\nW", W.shape)

# Epochs
n = 100

# Activation function hyp or sigmoid
activation = hyp
for epochs in range(n+1):
    # Activated weighted sum for forward pass
    o = np.array(activation(X * W))

    # Derivative of loss w.r.t weight
    E_W = X * activation(o, der=True) * loss_derivative(Y, o)

    # Gradient descent weight update rule for backward pass
    W = W - E_W
    # print("\nOutput at " + str(epochs) + "th epoch:" + "\n", o)

    # Saves regression line plot at each epoch and only shows the final plot
    plot(X, Y, o, 0, 1, save=False)

    # Results
    if epochs == n:
        print("Mean absolute percentage error: ", np.mean(abs(Y-o)*100), "%")
        print("Output:\n", DataFrame(Y).iloc[15:25, :])
        print("\nPrediction:\n", DataFrame(o).iloc[15:25, :])


# # Neural network for multi-dimensional data (Classification, incomplete)
# import numpy as np
# inputs = [1,2,3,2.5]                    #4x1
# weights = [[0.2,0.8,-0.5,1.0],          #3x4
#            [0.5,-0.91,0.26,-0.5],
#            [-0.26,-0.27,0.17,0.87]]
# biases = [2,3,0.5]                      #3x1
# # print(list(zip(weights,biases)))
# # print(list(zip(inputs, [neuron_weights for neuron_weights in weights])))
# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights,biases):
#     # print(neuron_weights)
#     # print(neuron_bias)
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         # print(n_input)
#         # print(weight)
#         neuron_output = neuron_output + n_input*weight
#     neuron_output = neuron_output + neuron_bias
#     layer_outputs.append(neuron_output)
# # print(np.array(layer_outputs))
# output = np.dot(weights, inputs) + biases
# # print(output)
#
# o = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
# t = np.array([0,1,1])
# """Take 0th array of o and then 0th element of o and so on"""
# pre_loss = o[list(range(len(t))),t]                                 # length of o and t are same cuz 3 samples
# loss = np.mean(-np.log(pre_loss))
# print(loss)
