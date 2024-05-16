import os
from io import open
import glob
import unicodedata
import string
import torch
import time
import math
import random
import yaml
# from model import (
#     RNNWordClassifier,
#     RNNWordGen
# )
files_path = 'data/names/*.txt'
with open('config.yaml') as config:
    config = yaml.safe_load(config)

n_hidden = config['train_config']['hyperparameters']['hidden_units']
category_words = {}
all_categories = []


def find_files(path: str) -> list[str]:
    """
    Find all files in a directory with a given path.

    Parameters:
        path (str): The path of the directory to search.

    Returns:
        list[str]: A list of file paths.
    """
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)


def unicode_to_ascii(s: str) -> str:
    """
    Convert Unicode string to ASCII, removing characters that are not in the
    ASCII character set or are in the Unicode Mn category.

    Args:
        s (str): The Unicode string to convert.

    Returns:
        str: The ASCII equivalent of the input string.
    """
    return ''.join(
        char
        for char in unicodedata.normalize("NFD", s)
        if unicodedata.category(char) != "Mn" and char in string.ascii_letters
    )


def read_words(file_name: str) -> list[str]:
    """
    Reads a text file and returns a list of words.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        List[str]: A list of words in the file.
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        words = file.read().strip().split('\n')
    return [unicode_to_ascii(word) for word in words]


for file_name in find_files(files_path):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    words = read_words(file_name)
    category_words[category] = words


n_categories = len(all_categories)


def letter_index(letter: str) -> int:
    """
    Returns the index of a given letter in the all_letters string.

    Args:
        letter (str): The letter to find the index of.

    Returns:
        int: The index of the given letter in the all_letters string.

    Raises:
        ValueError: If the given letter is not in the all_letters string.
    """
    if letter not in all_letters:
        raise ValueError(f"{letter} is not a valid letter.")
    return all_letters.index(letter)


def letter_to_tensor(letter: str) -> torch.Tensor:
    """
    Convert a single letter into a tensor representation.

    Args:
        letter (str): The letter to convert.

    Returns:
        torch.Tensor: The tensor representation of the letter.
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_index(letter)] = 1
    return tensor


def word_to_tensor(word: str) -> torch.Tensor:
    """
    Convert a word into a tensor representation.

    Args:
        word (str): The word to convert.

    Returns:
        torch.Tensor: The tensor representation of the word.
    """
    tensor = torch.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][letter_index(letter)] = 1
    return tensor


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


'''
Generator helpers
'''


def input_tensor(word):
    tensor = torch.zeros(len(word), 1, n_letters + 1)
    for index, letter in enumerate(word):
        tensor[index][0][letter_index(letter)] = 1
    return tensor


def category_tensor(category):
    index = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][index] = 1
    return tensor


def target_tensor(word):
    letter_indexes = [
        all_letters.find(word[index])
        for index in range(1, len(word))
        ]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


def random_choice(list_item):
    return list_item[random.randint(0, len(list_item) - 1)]


def random_training_pair():
    category = random_choice(all_categories)
    word = random_choice(category_words[category])
    return category, word


def random_training_example():
    _category, _word = random_training_pair()
    _category_tensor = category_tensor(_category)
    _input_tensor = input_tensor(_word)
    _target_tensor = target_tensor(_word)
    return _category_tensor, _input_tensor, _target_tensor


'''

'''


def predict_class(model, input_word, n_predictions=1):
    """
    Predict the class of an input word.

    Args:
        input_word (str): The input word.
        n_predictions (int, optional): The number of predictions to return.
            Defaults to 1.

    Returns:
        str: The predicted class.

    """
    word_tensor = word_to_tensor(input_word)
    hidden = model.init_hidden()
    model.eval()

    with torch.no_grad():
        for i in range(word_tensor.size()[0]):
            output, hidden = model(word_tensor[i], hidden)

        top_pred, top_pred_idx = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            pred = top_pred[0][i].item()
            category_idx = top_pred_idx[0][i].item()
            # print('(%.2f) %s' % (pred, all_categories[category_idx]))
            predictions.append([pred, all_categories[category_idx]])
    return predictions[0][1]


def generate_name(model, category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        _category_tensor = category_tensor(category)
        input = input_tensor(start_letter)
        hidden = model.init_hidden()

        output_name = start_letter
        max_length = 20
        for i in range(max_length):
            output, hidden = model(_category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)

        return output_name
