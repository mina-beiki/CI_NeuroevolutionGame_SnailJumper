import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.layer_sizes = layer_sizes
        input_layer_neurons = self.layer_sizes[0]
        hidden_layer_neurons = self.layer_sizes[1]
        output_neurons = self.layer_sizes[2]

        self.W_1 = np.random.randn(input_layer_neurons * hidden_layer_neurons).reshape(hidden_layer_neurons, input_layer_neurons)
        self.b_1 = np.zeros((hidden_layer_neurons, 1))
        self.W_2 = np.random.randn(output_neurons * hidden_layer_neurons).reshape(output_neurons, hidden_layer_neurons)
        self.b_2 = np.zeros((output_neurons, 1))

    def activation(self, x, function='sigmoid'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        # We can choose between sigmoid or softmax function:
        softmax_res = np.exp(x) / np.exp(x).sum()
        sigmoid_res = 1 / (1 + np.exp(-x))
        if function == 'softmax':
            return softmax_res
        else:
            return sigmoid_res

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        #convert input into array:
        x = np.array(x)
        x = x.reshape(self.layer_sizes[0], 1)
        #calculate z:
        z1 = self.W_1 @ x + self.b_1
        #calculate a (after using activation function):
        a1 = self.activation(z1)
        #calculate res of nn:
        res = self.activation(self.W_2 @ a1 + self.b_2, function='softmax')

        return res
