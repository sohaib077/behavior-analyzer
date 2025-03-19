from flask import Flask, request, jsonify
import joblib
import pandas as pd
from collections import defaultdict

# Load the trained model and encoder
model = joblib.load('app_usage_model.pkl')
encoder = joblib.load('app_encoder.pkl')

# Create a Flask app
app = Flask(__name__)


# Define a function to make predictions
def predict_activity(app_usage_dict):
    # Convert the input dictionary into a list of (app, minutes) tuples
    app_usage_list = list(app_usage_dict.items())

    # Convert the input list into a DataFrame
    input_data = []
    for app, minutes in app_usage_list:
        try:
            # Try to encode the app
            app_encoded = encoder.transform([[app]]).toarray()  # Use .toarray() to get a dense array
            app_encoded_df = pd.DataFrame(app_encoded)
            app_encoded_df.columns = app_encoded_df.columns.astype(str)  # Convert column names to strings
            input_data.append(pd.concat([app_encoded_df, pd.Series(minutes, name='minutes')], axis=1))
        except:
            # If the app is not in the encoder's vocabulary, categorize it as "Other"
            # Create a DataFrame with all zeros for the encoded features
            zero_features = {col: 0 for col in encoder.get_feature_names_out()}
            zero_df = pd.DataFrame([zero_features])
            zero_df['minutes'] = minutes
            input_data.append(zero_df)

    input_data = pd.concat(input_data)
    input_data.columns = input_data.columns.astype(str)  # Ensure all column names are strings

    # Make predictions
    predictions = model.predict(input_data)
    return predictions


# Define a function to generate a summarized report and behavioral indicator
def generate_report(app_usage_dict):
    predictions = predict_activity(app_usage_dict)
    report = defaultdict(int)
    for (app, minutes), category in zip(app_usage_dict.items(), predictions):
        if app not in encoder.categories_[0]:  # If the app is not in the encoder's vocabulary
            report['Other'] += minutes
        else:
            report[category] += minutes

    # Calculate the total time spent
    total_time = sum(report.values())

    # Calculate the percentage of time spent on each category
    report_percentages = {category: (time / total_time) * 100 for category, time in report.items()}

    # Define rules for behavioral indicator based on percentages
    learning_percent = report_percentages.get('Learning', 0)
    playing_percent = report_percentages.get('Games', 0)
    watching_percent = report_percentages.get('Watching Series', 0)
    music_percent = report_percentages.get('Music', 0)
    social_percent = report_percentages.get('Social Media', 0)
    other_percent = report_percentages.get('Other', 0)

    # Define rules for behavioral indicator
    if learning_percent > 50 and playing_percent < 20 and watching_percent < 20 and social_percent < 20 and other_percent < 20:
        behavior = "Very Good"
    elif learning_percent >= 30 >= watching_percent and playing_percent <= 30 and social_percent <= 30 and other_percent <= 30:
        behavior = "Good"
    elif learning_percent >= 10 or playing_percent <= 50 or watching_percent <= 50 or social_percent <= 50 or other_percent <= 50:
        behavior = "Bad"
    else:
        behavior = "Very Bad"

    # Add the behavioral indicator to the report
    report['Behavior'] = behavior

    # Add percentages to the report
    report['Percentages'] = report_percentages

    return report


# Define an API endpoint to receive data and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json

    # Example input format:
    # {
    #     "com.google.android.youtube": 20,
    #     "com.instagram.android": 50,
    #     "com": 10
    # }

    # Generate the report
    report = generate_report(data)

    # Return the report as a JSON response
    return jsonify(report)


# Add a test route for debugging
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)