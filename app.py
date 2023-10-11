from flask import Flask, render_template, request
import regression

app = Flask(__name__)

# Function to validate input and provide specific error messages
def is_valid_input(runs, wickets, overs):
    error_messages = []
    if runs < 0:
        error_messages.append("Runs should be greater than or equal to 0.")
    if wickets < 0 or wickets > 10:
        error_messages.append("Wickets should be between 0 and 10.")
    if overs < 0 or overs > 50:
        error_messages.append("Overs should be between 0 and 50.")
    
    # Check for valid overs format (0.1 to 0.6)
    if overs % 1 > 0.7:
        error_messages.append("Invalid overs format. Should be in the format of 0.1 to 0.6.")
    
    return not error_messages, error_messages


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    error_messages = []  # Create an empty list to store error messages

    try:
        runs = float(request.form['runs'])
        wickets = float(request.form['wickets'])
        overs = float(request.form['overs'])

        # Check if inputs are valid and get specific error messages
        is_valid, validation_errors = is_valid_input(runs, wickets, overs)

        if not is_valid:
            error_messages.extend(validation_errors)  # Add validation errors to the list
            error_messages.append("Please input valid values.")  # Add a general error message
        else:
            # Retrieve the selected model from the form data
            selected_model = request.form.get('model')
            if selected_model in ["RandomForest", "SVM", "Polynomial"]:
                prediction, errors = regression.predict_total(runs, wickets, overs, selected_model)

                return render_template('index.html', prediction=prediction, error=None, errors=errors)
            else:
                error_messages.append("Invalid model selection.")
                return render_template('index.html', error="<br>".join(error_messages))

    except ValueError:
        error_messages.append("Please enter valid numeric values for runs, wickets, and overs.")
        return render_template('index.html', error="<br>".join(error_messages))

if __name__ == '__main__':
    app.run(debug=True)
