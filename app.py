from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)
model = joblib.load('modelo_entrenado.pkl')

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Página principal con formulario
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    class_name = None

    if request.method == 'POST':
        try:
            # Obtener valores del formulario
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Hacer predicción
            features = [sepal_length, sepal_width, petal_length, petal_width]
            pred = model.predict([features])[0]
            prediction = int(pred)
            class_name = target_names[pred]
        except:
            prediction = "Error en los datos"

    # HTML directamente en el archivo (puedes separar luego si quieres)
    html = '''
    <html>
    <head>
        <title>Clasificador de Iris</title>
    </head>
    <body>
        <h1>Predicción de flor Iris 🌸</h1>
        <form method="post">
            <label>Largo del sépalo:</label><br>
            <input type="text" name="sepal_length"><br><br>
            <label>Ancho del sépalo:</label><br>
            <input type="text" name="sepal_width"><br><br>
            <label>Largo del pétalo:</label><br>
            <input type="text" name="petal_length"><br><br>
            <label>Ancho del pétalo:</label><br>
            <input type="text" name="petal_width"><br><br>
            <input type="submit" value="Predecir">
        </form>
        {% if prediction is not none %}
            <h2>Predicción: {{ class_name }} (clase {{ prediction }})</h2>
        {% endif %}
    </body>
    </html>
    '''

    return render_template_string(html, prediction=prediction, class_name=class_name)

if __name__ == '__main__':
    app.run(debug=True)
