from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the ColumnTransformer (Assuming it's saved as 'transformer.pkl')
with open("transformer.pkl", "rb") as transformer_file:
    transformer = pickle.load(transformer_file)

# Define the column names (Must match training data)
columns = ['property_type', 'location', 'city', 'baths', 'purpose', 'bedrooms', 'Area Type', 'Area Size']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        input_data = [request.form[col] for col in columns]
        input_df = pd.DataFrame([input_data], columns=columns)

        # Preprocess input using the same transformer used during training
        transformed_input = transformer.transform(input_df)

        # Make a prediction
        prediction = model.predict(transformed_input)[0]

        # Return result
        return render_template('index.html', prediction_text=f'Predicted Price: {prediction:.2f}')

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


       

    
        




