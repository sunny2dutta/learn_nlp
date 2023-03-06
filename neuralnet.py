import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()


     def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        return x
