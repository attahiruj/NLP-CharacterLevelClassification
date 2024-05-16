from flask import Flask, jsonify, request
import torch
from utils import (
    generate_name,
    predict_class,
    n_letters,
    n_hidden,
    n_categories
    )
from model import (
    RNNWordClassifier,
    RNNWordGenerator
)

classifier = RNNWordClassifier(n_letters, n_hidden, n_categories)
generator = RNNWordGenerator(n_letters+1, n_hidden, n_letters+1)
classifier.load_state_dict(torch.load('./models/name_classifier.pt'))
generator.load_state_dict(torch.load('./models/name_generator.pt'))

app = Flask(__name__)


@app.route('/')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        json = request.get_json()
        prediction = predict_class(classifier, json['name'])
        return jsonify({'language': prediction})
    else:
        return 'Hello World!'


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        json = request.get_json()
        prediction = generate_name(generator, json['language'], json['letter'])
        return jsonify({'name': prediction})
    else:
        return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
