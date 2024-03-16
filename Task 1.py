import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('Dataset/HousePricePrediction.csv')

# Print the first five rows of the dataset
print(data.head())

# Drop the 'Id' column as it's unnecessary
data.drop(['Id'], axis=1, inplace=True)

# Fill missing values in the 'SalePrice' column with the mean
data['SalePrice'] = data['SalePrice'].fillna(data['SalePrice'].mean())

# Print the shape of the dataset
print(data.shape)

# Print the column names
print(data.columns)

# Print the information about the dataset
print(data.info())

# Descriptive statistics of the dataset
print(data.describe())

# Identify categorical, integer, and float variables
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Seaborn for data visualization
sns.pairplot(data)

# Visualize the number of unique values of categorical features
unique_values = [data[col].nunique() for col in object_cols]
plt.figure(figsize=(10, 6))
plt.title('Number of Unique Values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)

# Drop rows with missing values
new_dataset = data.dropna()

# Check for remaining missing values
print(new_dataset.isnull().sum())

# Reidentify categorical variables
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('Number of categorical features:', len(object_cols))

# One-hot encode categorical variables
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Separate independent and dependent variables
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict the target variable on the test set
predictions = model.predict(X_test)

# Visualize predicted vs actual values using scatter plot
plt.scatter(Y_test, predictions)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')

# Statistical operations
def mean_squared_error(real, pred):
    return np.mean((pred - real) ** 2)

def mean_absolute_error(real, pred):
    return np.mean(np.abs(pred - real))

def root_mean_squared_error(real, pred):
    return np.sqrt(np.mean((pred - real) ** 2))

def variance_inflation_factor(real, pred):
    n = len(real)
    m = np.mean(real)
    return np.sum((real - m) ** 2) / (n - pred)

# Calculate evaluation metrics
print("Mean Squared Error:", mean_squared_error(Y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(Y_test, predictions))
print("Root Mean Squared Error:", root_mean_squared_error(Y_test, predictions))
print("Variance Error:", variance_inflation_factor(Y_test, predictions))
