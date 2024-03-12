from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# @app.route("/", methods=['POST'])
@app.post('/run')
def predict():
    data = request.get_json()

    vars = [0] * 10

    vars[0] = data["sex"]
    vars[1] = data["age"]
    vars[2] = data["education"]
    vars[3] = data["smoker"]
    vars[4] = data["cigs"]
    vars[5] = data["bpMeds"]
    vars[6] = data["stroke"]
    vars[7] = data["hypertensia"]
    vars[8] = data["diabetes"]
    vars[9] = data["bmi"]

    vars = np.array(vars).reshape(1,-1)

    with open("knn.pkl", "rb") as file:
        model = pickle.load(file)

    result = dict()

    classification = model.predict(vars)
    proba = model.predict_proba(vars)[:,1]

    result["classification"] = int(classification[0])
    result["probability"] = float(proba)

        # result.headers.add('Access-Control-Allow-Origin', '*')

    return result, 200
    
    # except Exception as error:
    #     return ("Error %s" % error), 500