from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

app = Flask(__name__) 

# Define the directory containing your pickle files
models_directory = 'models'

# Assuming you have six pipeline files, one for each model
pipeline_files = {
    '1': 'pipeline1.pkl',
    '2': 'pipeline2.pkl',
    '3': 'pipeline3.pkl',
    '4': 'pipeline4.pkl',
    '5': 'pipeline5.pkl',
    '6': 'pipeline6.pkl',
}

# Load all models into a dictionary
models = {target: joblib.load(os.path.join(models_directory, pipeline_file)) for target, pipeline_file in pipeline_files.items()}

# Load the training data
training_data = pd.read_csv('my_data.csv')

# Create a ColumnTransformer for preprocessing
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = SimpleImputer(strategy='most_frequent')

# Identify numeric and categorical columns
numeric_features = ['grade_10', 'grade_12', 'graduation_score', 'admission_test_score', 'work_exp_months']
categorical_features = ['gender', 'category', 'graduation_type']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit the ColumnTransformer on the training data
preprocessor.fit(training_data)

# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    gender = 1 if request.form['gender'] == 'male' else 0  # Convert gender to numeric
    category = request.form['category']
    grade10 = float(request.form['grade10'])
    grade12 = float(request.form['grade12'])
    graduation_score = float(request.form['graduationScore'])
    graduation_type = 0 if request.form['graduationType'] == 'Engineer' else 1  # Convert graduation_type to numeric
    admission_test_score = float(request.form['admissionTestScore'])
    work_experience = float(request.form['workExperience'])

    # Combine user inputs into a DataFrame for testing
    user_input_data = pd.DataFrame({
        'gender': [gender],
        'category': [category],
        'grade_10': [grade10],
        'grade_12': [grade12],
        'graduation_score': [graduation_score],
        'graduation_type': [graduation_type],
        'admission_test_score': [admission_test_score],
        'work_exp_months': [work_experience]
    })

    # Apply the preprocessing steps
    user_input_data = preprocessor.transform(user_input_data)

    # Convert the transformed user_input_data to a DataFrame
    user_input_data = pd.DataFrame(user_input_data, columns=numeric_features + categorical_features)

    # Perform predictions using each model
    predictions = {}
    for target, model in models.items():
        prediction_proba = model.predict_proba(user_input_data)
        predicted_class = model.predict(user_input_data)[0]

        predictions[target] = {
            'predicted_class': predicted_class,
        }

    # You can now use 'predictions' in your template
    return render_template('predict.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
