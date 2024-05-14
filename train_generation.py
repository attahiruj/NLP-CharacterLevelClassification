from utils import (
    category_words,
    all_categories,
    n_categories,
    n_letters,
    word_to_tensor)

from model import RNNWordGen
import random
import time
import math
import torch
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def random_training_example() -> tuple[str, str, torch.Tensor, torch.Tensor]:
    """
    Returns a random training example from the dataset.

    Returns:
        Tuple[str, str, torch.Tensor, torch.Tensor]: A random training example
        consisting of the category, word, category tensor, and word tensor.

    """
    category = random_choice(all_categories)
    word = random_choice(category_words[category])
    category_tensor = torch.tensor(
                                    [all_categories.index(category)],
                                    dtype=torch.long
                                    )

    word_tensor = word_to_tensor(word)
    return category, word, category_tensor, word_tensor


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


def evaluate(word_tensor: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the RNN on a given word tensor.

    Args:
        word_tensor (torch.Tensor): The word tensor of shape (sequence_length).

    Returns:
        torch.Tensor: The output tensor of shape (output_size, ).

    """
    hidden = rnn.init_hidden()

    for i in range(len(word_tensor[0])):
        output, hidden = rnn(word_tensor[i], hidden)

    return output


def time_since(since):
    """
    Returns a string with the time difference since the input time since value.

    Args:
        since (float): The time value to calculate the difference from.

    Returns:
        str: The time difference as a string.
    """
    now = time.time()
    sec = now - since
    min = math.floor(sec / 60)
    sec -= min * 60
    return f"{min}m {sec:.0f}s"


rnn = RNNWordGen(n_letters, n_hidden, n_letters)

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
        print('%d\t %d%% (%s) %.4f' % (iter, iter / n_iters * 100, time_since(start), loss))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

print('\n')

save_model = config['train_config']['save_model']
if save_model[0]:
    model_path = 'models/name_generator.pt'
    torch.save(rnn.state_dict(), save_model[1])
    print('model saved to: {}'.format(model_path))

plot_losses = config['train_config']['plot_losses']
if plot_losses:
    plt.figure()
    plt.plot(all_losses)

plot_confusion_matrix = config['train_config']['plot_confusion_matrix']
if plot_confusion_matrix:
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # record correct guesses from examples
    for i in range(n_confusion):
        category, word, category_tensor, word_tensor = random_training_example()

        output = evaluate(word_tensor)
        guess, guess_idx = category_from_output(output)
        category_idx = all_categories.index(category)
        confusion[category_idx][guess_idx] += 1

    # normalize confusion
    for i in range(n_categories):
        confusion[i] = confusion[i]/confusion[i].sum()

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
