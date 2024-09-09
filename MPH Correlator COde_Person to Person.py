import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Path to your .db file
db_path = 'C:\\Users\\Joey\\PycharmProjects\\Machine Learning\\Data\\grading_equation_reference_data_PtoP.db'

# Establishing a connection to the database
conn = sqlite3.connect(db_path)
# Get the list of all columns in the 'variables' table
column_query = "PRAGMA table_info(variables)"
columns_info = pd.read_sql_query(column_query, conn)

# Assuming the first column is an ID or similar, and 'MPH' is the target variable
# You want to include 'MPH' explicitly and then add columns 4 through 22 by their names
# Note: Adjust the indices 3:22 based on the actual positions of your desired columns
desired_columns = ['MPH'] + columns_info.loc[5:21, 'name'].tolist()

# Constructing the SQL query with the desired columns
query = f"SELECT {', '.join(desired_columns)} FROM variables"

# Reading the data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Remove rows where MPH is NaN
df = df.dropna(subset=['MPH'])

# Handling non-numeric data: Convert categorical variables using one-hot encoding
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Splitting the dataset
X = df.drop('MPH', axis=1)
y = df['MPH'].valuesgit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)