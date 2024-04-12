from utils import * 
import torch.nn as nn
import torch.nn.functional as F
import yaml

with open('config.yaml') as config:
    config = yaml.safe_load(config)

n_hidden = config['train_config']['hyperparameters']['hidden_units']

class RNN(nn.Module):
	"""
	A simple RNN model with fully connected layers.

	Args:
		input_size (int): The number of input features.
		hidden_size (int): The number of hidden units.
		output_size (int): The number of output classes.

	Attributes:
		hidden_size (int): The number of hidden units.
		i2h (torch.nn.Linear): The linear layer for input to hidden connections.
		h2h (torch.nn.Linear): The linear layer for hidden to hidden connections.
		h2o (torch.nn.Linear): The linear layer for hidden to output connections.
		softmax (torch.nn.LogSoftmax): The softmax layer for output activation.

	"""

	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size, hidden_size)
		self.h2h = nn.Linear(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		"""
		Forward pass of the RNN.

		Args:
			input (torch.Tensor): The input tensor of shape (batch_size, input_size).
			hidden (torch.Tensor): The hidden state tensor of shape (1, hidden_size).

		Returns:
			torch.Tensor: The output tensor of shape (batch_size, output_size)
			torch.Tensor: The updated hidden state tensor of shape (1, hidden_size)

		"""
		hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
		output = self.h2o(hidden)
		output = self.softmax(output)
		return output, hidden

	def init_hidden(self):
		"""
		Initialize the hidden state tensor.

		Returns:
			torch.Tensor: The initialized hidden state tensor of shape (1, hidden_size)

		"""
		return torch.zeros(1, self.hidden_size)

model = config['train_config']['save_model'][1]
rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load(model))
def evaluate(word_tensor):
	hidden = rnn.init_hidden()
	
	for i in range(word_tensor.size()[0]):
		output, hidden = rnn(word_tensor[i], hidden)
	
	return output


def predict_class(input_word, n_predictions=1):
    # print('\n> %s' % input_word)
    with torch.no_grad():
        output = evaluate(word_to_tensor(input_word))
        top_pred, top_pred_idx = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            pred = top_pred[0][i].item()
            category_idx = top_pred_idx[0][i].item()
            # print('(%.2f) %s' % (pred, all_categories[category_idx]))
            predictions.append([pred, all_categories[category_idx]])
    return predictions[0][1]
