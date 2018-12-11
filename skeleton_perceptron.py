import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.process(training_inputs)

            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_der(output))

            self.synaptic_weights += adjustments

    def process(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == '__main__':
    neural_network = NeuralNetwork()
    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    training_iterations = 10000

    neural_network.train(training_inputs, training_outputs, training_iterations)

    print('Synaptic Weights after training are:')
    print(neural_network.synaptic_weights)

    inp1 = str(input('Input A:'))
    inp2 = str(input('Input B:'))
    inp3 = str(input('Input C:'))
    out = neural_network.process(np.array([inp1, inp2, inp3]))
    print('Test Case:')
    print('Input data', [inp1, inp2, inp3])
    print('Output', int(round(out[0])))
