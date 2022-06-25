from copyreg import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Read scv-file
data = pd.read_csv('data/auto-mpg.csv', sep=";")

# print(data)

# shuffle data
data = data.sample(frac=1)

# "class"-column
y_variable = data['mpg']

# all columns that are not 'class'-columns that contain the atributes
x_variables = data.loc[:, data.columns != 'mpg']

x_train, x_test, y_train, y_test = train_test_split(
    x_variables, y_variable, test_size=0.2)

regressor = LinearRegression()

regressor = regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
# print(y_pred)

file_to_write = open("data/models/baummethoden_lr.prickle", "wb")
pickle.dump(regressor, file_to_write)
