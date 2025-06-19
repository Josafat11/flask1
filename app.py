from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('modelo_entrenado.pkl')

@app.route('/')
def home():
    return "¡Hola, Mundo!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
