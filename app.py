from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('modelo_entrenado.pkl')

# Diccionario para convertir números a nombres
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

@app.route('/')
def home():
    return "¡Hola, Mundo!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    predicted_class = target_names[prediction[0]]
    return jsonify({
        'prediction': int(prediction[0]),
        'class_name': predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)
