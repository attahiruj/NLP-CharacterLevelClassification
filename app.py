from flask import Flask, jsonify, request
import torch
from model import (
    classifier,
    model
)
from utils import (
    all_categories,
    word_to_tensor,
    evaluate_classifier
    )


app = Flask(__name__)


classifier.load_state_dict(torch.load(model))


def predict_class(input_word, n_predictions=1):
    """
    Predict the class of an input word.

    Args:
        input_word (str): The input word.
        n_predictions (int, optional): The number of predictions to return.
            Defaults to 1.

    Returns:
        str: The predicted class.

    """
    # print('\n> %s' % input_word)
    with torch.no_grad():
        output = evaluate_classifier(word_to_tensor(input_word), classifier)
        top_pred, top_pred_idx = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            pred = top_pred[0][i].item()
            category_idx = top_pred_idx[0][i].item()
            # print('(%.2f) %s' % (pred, all_categories[category_idx]))
            predictions.append([pred, all_categories[category_idx]])
    return predictions[0][1]


def generate_name(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        _category_tensor = category_tensor(category)
        input = input_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(_category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)

        return output_name

@app.route('/')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        json = request.get_json()
        print(json['name'])
        prediction = predict_class(json['name'])
        return jsonify({'language': prediction})
    else:
        return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
