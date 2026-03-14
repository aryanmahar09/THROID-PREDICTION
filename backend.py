from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved model, preprocessor, and label encoder
try:
    with open(os.path.join(BASE_DIR, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("✗ Error: best_model.pkl not found. Run the notebook first to generate it.")
    exit()

try:
    with open(os.path.join(BASE_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    print("✓ Preprocessor loaded successfully")
except FileNotFoundError:
    print("✗ Error: preprocessor.pkl not found. Run the notebook first to generate it.")
    exit()

try:
    with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    print("✓ Label encoder loaded successfully")
except FileNotFoundError:
    print("✗ Error: label_encoder.pkl not found. Run the notebook first to generate it.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = float(data.get('age', 0))
        sex = float(data.get('sex', 0))
        on_thyroxine = float(data.get('on_thyroxine', 0))
        query_on_thyroxine = float(data.get('query_on_thyroxine', 0))
        on_antithyroid_medication = float(data.get('on_antithyroid_medication', 0))
        sick = float(data.get('sick', 0))
        pregnant = float(data.get('pregnant', 0))
        thyroid_surgery = float(data.get('thyroid_surgery', 0))
        i131_treatment = float(data.get('i131_treatment', 0))
        query_hypothyroid = float(data.get('query_hypothyroid', 0))
        query_hyperthyroid = float(data.get('query_hyperthyroid', 0))
        lithium = float(data.get('lithium', 0))
        goitre = float(data.get('goitre', 0))
        tumor = float(data.get('tumor', 0))
        hypopituitary = float(data.get('hypopituitary', 0))
        psych = float(data.get('psych', 0))
        tsh_measured = float(data.get('tsh_measured', 0))
        tsh = float(data.get('tsh', 0))
        t3_measured = float(data.get('t3_measured', 0))
        tt4_measured = float(data.get('tt4_measured', 0))
        tt4 = float(data.get('tt4', 0))
        t4u_measured = float(data.get('t4u_measured', 0))
        t4u = float(data.get('t4u', 0))
        fti_measured = float(data.get('fti_measured', 0))
        fti = float(data.get('fti', 0))
        
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'on thyroxine': [on_thyroxine],
            'query on thyroxine': [query_on_thyroxine],
            'on antithyroid medication': [on_antithyroid_medication],
            'sick': [sick],
            'pregnant': [pregnant],
            'thyroid surgery': [thyroid_surgery],
            'I131 treatment': [i131_treatment],
            'query hypothyroid': [query_hypothyroid],
            'query hyperthyroid': [query_hyperthyroid],
            'lithium': [lithium],
            'goitre': [goitre],
            'tumor': [tumor],
            'hypopituitary': [hypopituitary],
            'psych': [psych],
            'TSH measured': [tsh_measured],
            'TSH': [tsh],
            'T3 measured': [t3_measured],
            'TT4 measured': [tt4_measured],
            'TT4': [tt4],
            'T4U measured': [t4u_measured],
            'T4U': [t4u],
            'FTI measured': [fti_measured],
            'FTI': [fti]
        })
        
        input_processed = preprocessor.transform(input_data)
        pred = model.predict(input_processed)
        result = le.inverse_transform(pred)[0]
        
        return jsonify({'prediction': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)