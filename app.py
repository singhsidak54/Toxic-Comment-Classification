import pickle
from model.tcc_model import MEModelNB
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def predict_form():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def predict():

    comment = request.form['comment']

    pred = clf.predict(comment)

    return jsonify({'class': pred})
#dummy commit
#test
if __name__ == '__main__':
    model = open('model/model.pkl', 'rb')

    clf = pickle.load(model)
    app.run(debug=True)
