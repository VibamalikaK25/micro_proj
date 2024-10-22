from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load models
rf_model = joblib.load('random_forest_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

# Load and preprocess data
df = pd.read_csv('water_quality.csv')
df.replace('#NUM!', np.nan, inplace=True)
df.dropna(inplace=True)
le = LabelEncoder()
df['is_safe'] = le.fit_transform(df['is_safe'])
X = df.drop('is_safe', axis=1)
y = df['is_safe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def get_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    try:
        loss = log_loss(y_test, model.predict_proba(X_test))
    except:
        loss = np.nan  # Handle cases where log_loss cannot be computed
    report = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions)
    return accuracy, loss, report, conf_matrix

# Calculate metrics
rf_accuracy, rf_loss, rf_report, rf_conf_matrix = get_metrics(rf_model, X_test, y_test)
dt_accuracy, dt_loss, dt_report, dt_conf_matrix = get_metrics(dt_model, X_test, y_test)

# Attribute descriptions
descriptions = {
    'aluminium': 'Dangerous if greater than 2.8',
    'ammonia': 'Dangerous if greater than 32.5',
    'arsenic': 'Dangerous if greater than 0.01',
    'barium': 'Dangerous if greater than 2',
    'cadmium': 'Dangerous if greater than 0.005',
    'chloramine': 'Dangerous if greater than 4',
    'chromium': 'Dangerous if greater than 0.1',
    'copper': 'Dangerous if greater than 1.3',
    'flouride': 'Dangerous if greater than 1.5',
    'bacteria': 'Dangerous if greater than 0',
    'viruses': 'Dangerous if greater than 0',
    'lead': 'Dangerous if greater than 0.015',
    'nitrates': 'Dangerous if greater than 10',
    'nitrites': 'Dangerous if greater than 1',
    'mercury': 'Dangerous if greater than 0.002',
    'perchlorate': 'Dangerous if greater than 56',
    'radium': 'Dangerous if greater than 5',
    'selenium': 'Dangerous if greater than 0.5',
    'silver': 'Dangerous if greater than 0.1',
    'uranium': 'Dangerous if greater than 0.3',
    'is_safe': 'Class attribute {0 - not safe, 1 - safe}'
}

@app.route('/')
def index():
    return render_template('dashboard.html', 
                           rf_accuracy=rf_accuracy,
                           dt_accuracy=dt_accuracy,
                           rf_loss=rf_loss,
                           dt_loss=dt_loss,
                           rf_conf_matrix=rf_conf_matrix.tolist(),
                           dt_conf_matrix=dt_conf_matrix.tolist(),
                           columns=X.columns)
@app.route('/input', methods=['GET', 'POST'])
def input_form():
    return render_template('input_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    aluminium = float(request.form['aluminium'])
    ammonia = float(request.form['ammonia'])
    arsenic = float(request.form['arsenic'])
    barium = float(request.form['barium'])
    cadmium = float(request.form['cadmium'])
    chloramine = float(request.form['chloramine'])
    chromium = float(request.form['chromium'])
    copper = float(request.form['copper'])
    flouride = float(request.form['flouride'])
    bacteria = float(request.form['bacteria'])
    viruses = float(request.form['viruses'])
    lead = float(request.form['lead'])
    nitrates = float(request.form['nitrates'])
    nitrites = float(request.form['nitrites'])
    mercury = float(request.form['mercury'])
    perchlorate = float(request.form['perchlorate'])
    radium = float(request.form['radium'])
    selenium = float(request.form['selenium'])
    silver = float(request.form['silver'])
    uranium = float(request.form['uranium'])

    # Create input array for the model
    input_data = np.array([[aluminium, ammonia, arsenic, barium, cadmium, chloramine,
                            chromium, copper, flouride, bacteria, viruses, lead,
                            nitrates, nitrites, mercury, perchlorate, radium,
                            selenium, silver, uranium]])

    # Use Random Forest model to predict
    prediction = rf_model.predict(input_data)

    # Prepare result message
    result = 'Water is safe to drink' if prediction[0] == 1 else 'Water is not safe to drink'

    # Render result page
    return render_template('result.html', prediction=result, descriptions=descriptions)

@app.route('/description')
def description():
    return render_template('description.html', descriptions=description)

if __name__ == '__main__':
    app.run(debug=True)
