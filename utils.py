import os
from io import open
import glob
import unicodedata
import string
import torch
import time
import math
# from model import RNN_name_classifier, 

files_path = 'data/names/*.txt'


def find_files(path: str) -> list[str]:
    """
    Find all files in a directory with a given path.

    Parameters:
        path (str): The path of the directory to search.

    Returns:
        list[str]: A list of file paths.
    """
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
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


category_words = {}
all_categories = []


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


def evaluate_classifier(word_tensor: torch.Tensor, model) -> torch.Tensor:
    """
    Evaluate the RNN model on a given input tensor.

    Args:
        word_tensor (torch.Tensor): input tensor of
        shape (batch_size, n_letters)

    Returns:
        torch.Tensor: output tensor of shape (batch_size, n_categories)

    """
    hidden = model.init_hidden()
    model.eval()
    for i in range(word_tensor.size()[0]):
        output, hidden = model(word_tensor[i], hidden)

    return output