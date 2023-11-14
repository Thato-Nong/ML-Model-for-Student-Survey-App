# Step 1: Data Collection and Preparation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv('student_survey_data_generated.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Convert categorical data to numerical representations
data_encoded = pd.get_dummies(data_filled, columns=['gender', 'major'])

# Normalize numerical data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_encoded)

# Split the data into training and testing sets
X = data_normalized.drop('happiness', axis=1)
y = data_normalized['happiness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Engineering (if needed)
# ... (Add relevant feature engineering steps here)

# Step 3: Model Selection and Training
from sklearn.linear_model import LinearRegression

# Choose a model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Step 4: Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 5: Model Deployment and Interpretation
# ... (Add deployment and interpretation steps here)
