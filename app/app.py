from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return "flask in online"

@app.route('/predict', methods=['POST'])
def position():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['MAC1'], data['MAC2'],data['MAC3'])]])
    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 5000, debug = True)

