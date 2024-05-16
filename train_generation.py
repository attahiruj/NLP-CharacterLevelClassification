from utils import (
    all_categories,
    n_letters,
    time_since,
    random_training_example)

from model import RNNWordGenerator
import random
import time
import torch
import torch.nn as nn
from torchinfo import summary
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.full_load(f)

# hyperparameters

n_hidden = config['train_config']['hyperparameters']['hidden_units']
learning_rate = config['train_config']['hyperparameters']['learning_rate']
n_iters = config['train_config']['hyperparameters']['iteration']
print_every = 5*n_iters//100
plot_every = print_every//5
current_loss = 0
all_losses = []
criterion = nn.NLLLoss()
n_letters = n_letters + 1


def category_from_output(output: torch.Tensor):
    """
    Returns the most likely category and its index from the output tensor.

    Args:
        output (torch.Tensor): The output tensor
        of shape (batch_size, output_size).

    Returns:
        Tuple[str, int]: The most likely category and its index.

    """
    top_num, top_num_idx = output.topk(1)
    category_i = top_num_idx[0].item()
    return all_categories[category_i], category_i


def random_choice(length: list):
    """
    Returns a random element from a list.

    Args:
        length (List[Any]): The list from which an element is to be chosen.

    Returns:
        Any: A random element from the list.

    """
    return length[random.randint(0, len(length) - 1)]


def train(
            category_tensor: torch.Tensor,
            input_tensor: torch.Tensor,
            target_tensor: torch.Tensor
            ) -> tuple[torch.Tensor, float]:
    """
    Trains the RNN on a single training example.

    Args:
        category (str): The category of the training example.
        word_tensor (torch.Tensor): The word tensor of the training example.

    Returns:
        tuple[torch.Tensor, float]: The output tensor and the loss value.

    """
    target_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()

    rnn.zero_grad()
    losses = torch.Tensor([0])

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_tensor[i], hidden)
        loss = criterion(output, target_tensor[i])
        losses += loss

    losses.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, losses.item() / input_tensor.size(0)


rnn = RNNWordGenerator(n_letters, n_hidden, n_letters)

print_model_summary = config['train_config']['print_model_summary']
if print_model_summary:
    print(summary(rnn))

total_loss = 0
all_losses = []
start = time.time()
for iter in range(1, n_iters+1):
    output, loss = train(*random_training_example())
    total_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        print('%d\t %d%% (%s) %.4f' % (iter,
                                       iter / n_iters * 100,
                                       time_since(start),
                                       loss))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

print('\n')

save_model = config['train_config']['save_model']
if save_model:
    model_path = 'models/name_generator.pt'
    torch.save(rnn.state_dict(), model_path)
    print('model saved to: {}'.format(model_path))
