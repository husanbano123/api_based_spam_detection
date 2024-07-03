from flask import Flask, render_template, request
import joblib

# Load the model
model = joblib.load('spam_classifier_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = model.predict([message])[0]
        return render_template('result.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
