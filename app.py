from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('iris.pkl', 'rb'))

app = Flask(__name__)

# ✅ Add Home Route
@app.route('/')
def home():
    return render_template('home.html')

# ✅ Fix the Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])

    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)[0]

    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
