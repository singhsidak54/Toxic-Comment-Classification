import pickle
from tcc_model import MEModelNB
from flask import Flask, request, jsonify

app = Flask(__name__)
model = open('model.pkl', 'rb')

clf = pickle.load(model)

@app.route("/predict", methods=["POST"])
def predict():

    comment = request.json['comment']

    pred = clf.predict(comment)

    return jsonify({'class': pred})

if __name__ == '__main__':

    app.run(debug=True)