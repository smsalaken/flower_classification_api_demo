from flask import Flask, render_template, request
import pandas as pd
import sklearn
import numpy as np
import joblib

# Create the application instance
app = Flask(__name__)

# load model
def load():
    global knn
    knn = joblib.load('models/knn_classifier.pkl')

# Create a URL route in our application for "/"
@app.route('/flower')
def predict():
    """
    This function just responds to the browser ULR
    localhost:5000/

    :return:        the rendered template 'home.html'
    """
    sl = request.args.get('sl')
    sw = request.args.get('sw')
    pl = request.args.get('pl')
    pw = request.args.get('pw')

    inp = pd.DataFrame([[sl, sw, pl, pw]], columns=['sl','sw','pl','pw'])

    predicted_class = knn.predict(inp)[0]

    return predicted_class

# If we're running in stand alone mode, run the application
if __name__ == '__main__':

    load()
    app.run(host='0.0.0.0')