from flask import Flask, jsonify, request
from model import *


app = Flask(__name__)

@app.route('/')
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
	if request.method == 'POST':
		json = request.get_json()
		print(json['name'])
		prediction = predict_class(json['name'])
		return jsonify({'language' : prediction})
	else:
		return 'Hello World!'

if __name__ == '__main__':
	app.run(debug=True)
