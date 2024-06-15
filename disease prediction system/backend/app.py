from flask import Flask, request, render_template
from model import predict_disease
from utils import preprocess_input

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = preprocess_input(data)
    prediction = predict_disease(features)
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
