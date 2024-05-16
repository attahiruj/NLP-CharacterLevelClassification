from utils import (
    n_categories,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNWordClassifier(nn.Module):
    """
    A simple RNN model with fully connected layers.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        output_size (int): The number of output classes.

    Attributes:
        hidden_size (int): The number of hidden units.
        i2h (torch.nn.Linear): linear layer for input to hidden connections.
        h2h (torch.nn.Linear): linear layer for hidden to hidden connections.
        h2o (torch.nn.Linear): linear layer for hidden to output connections.
        softmax (torch.nn.LogSoftmax): The softmax layer for output activation.

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNWordClassifier, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        Forward pass of the RNN.

        Args:
            input (torch.Tensor): input tensor, shape (batch_size, input_size).
            hidden (torch.Tensor): hidden state tensor, shape (1, hidden_size).

        Returns:
            torch.Tensor: output tensor of shape (batch_size, output_size)
            torch.Tensor: updated hidden state tensor of shape (1, hidden_size)

        """
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state tensor.

        Returns:
            torch.Tensor: The initialized hidden state tensor
            of shape (1, hidden_size)

        """
        return torch.zeros(1, self.hidden_size)


class RNNWordGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNWordGenerator, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
                             hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size,
                             output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
