import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Path to your .db file
db_path = 'C:\\Users\\Joey\\PycharmProjects\\Machine Learning\\Data\\grading_equation_reference_data_all.db'

# Establishing a connection to the database
conn = sqlite3.connect(db_path)

# Get the list of all columns in the 'variables' table
column_query = "PRAGMA table_info(variables)"
columns_info = pd.read_sql_query(column_query, conn)

# Assuming the first column is an ID or similar and you want columns 4 through 22 (adjust as needed)
# Column indexes in the DataFrame start from 0, so for columns 4 through 22 you actually want indexes 3 through 21
desired_columns = columns_info.loc[3:21, 'name'].tolist()

# Adding 'MPH' to the beginning of the list if it's not already included and you need it for your analysis
if 'MPH' not in desired_columns:
    desired_columns.insert(0, 'MPH')

# Constructing the SQL query with the desired columns
query = f"SELECT {', '.join(desired_columns)} FROM variables"

# Reading the data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Now, df contains your data, and you can proceed with data preparation and model training

# Assuming you have already loaded your dataframe 'df' from the database

# Handling NaN values in MPH - choose one of the methods mentioned above
df = df.dropna(subset=['MPH'])  # Method 1: Dropping rows with NaN in MPH
# OR
df['MPH'] = df['MPH'].fillna(df['MPH'].median())  # Method 2: Imputation

# Assuming df is loaded and contains both numeric and non-numeric data

# Splitting the dataset into training and testing sets first
X = df.drop('MPH', axis=1)  # Predictor variables, before removing or encoding non-numeric columns
y = df['MPH']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying non-numeric columns in the training set
non_numeric_columns_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Option 1: Drop non-numeric columns from both training and testing sets
X_train = X_train.drop(columns=non_numeric_columns_train)
X_test = X_test.drop(columns=non_numeric_columns_train)

# Option 2: Convert categorical variables to numeric
# Note: For one-hot encoding, you need to ensure that the same dummy variables are present in both training and testing sets.
# This might require adjusting after encoding if the train and test sets have different categories.
# This is a more complex scenario that often requires careful alignment of columns after encoding.

# Proceed with model training using the adjusted X_train and X_test
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluations
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Adjusting the line for creating the coefficients DataFrame to use X_train.columns
coefficients = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficient'])
print(coefficients)

