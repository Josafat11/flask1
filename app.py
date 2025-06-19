from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('modelo_entrenado.pkl')

@app.route('/')
def home():
    return "¡Hola, Mundo!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validación básica
    if not data or 'features' not in data:
        return jsonify({'error': 'Debes enviar un JSON con la clave "features"'}), 400

    features = data['features']

    if len(features) != 4:
        return jsonify({'error': 'El vector de características debe tener 4 valores'}), 400

    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
