from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Cargar datos de ejemplo
X, y = load_iris(return_X_y=True)

# Entrenar modelo
model = RandomForestClassifier()
model.fit(X, y)

# Guardar modelo entrenado
joblib.dump(model, 'modelo_entrenado.pkl')
