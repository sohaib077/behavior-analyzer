import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the updated dataset
data = pd.read_csv('app_usage_data.csv')

# Preprocess the data
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_apps = encoder.fit_transform(data[['app_package_name']]).toarray()

# Combine the encoded apps and minutes into a single feature set
X = pd.concat([pd.DataFrame(encoded_apps), data['minutes']], axis=1)
X.columns = X.columns.astype(str)  # Ensure all column names are strings

y = data['category']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the updated model and encoder
joblib.dump(model, 'app_usage_model.pkl')
joblib.dump(encoder, 'app_encoder.pkl')

print("Model retrained and saved successfully!")