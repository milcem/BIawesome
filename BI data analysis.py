import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# open and read the csv file
df = pd.read_csv('menu_info.csv')
# print head of DataFrame
print(df.head())
# define the columns
columns = ['Restaurant', 'Dish_name', 'Preparation_duration', 'Waiting_duration', 'Price', 'Consumption_duration',
           'Serving_duration', 'Spice_density', 'Personnel_needed']
#print(df.head())



# provide basic statistics for data
print(df.describe())
# info on types of vars
print(df.info())



# one-hot encoding the categorical variables
one_hot_encoded_data = pd.get_dummies(df, columns=['Dish_name', 'Restaurant'])
print(one_hot_encoded_data)



# check covariance among variables
corr1 = one_hot_encoded_data.corr()
print(corr1[corr1 < 1].unstack().transpose() \
      .sort_values(ascending=False) \
      .drop_duplicates())

# drop Waiting_duration, and Price as has highest covariance with other variables
one_hot_encoded_data = one_hot_encoded_data.drop(['Waiting_duration', 'Price'], axis=1)
print(one_hot_encoded_data.describe())

corr2 = one_hot_encoded_data.corr()
print(corr2[corr2 < 1].unstack().transpose() \
      .sort_values(ascending=False) \
      .drop_duplicates())



# define x and y variables for the ml train/test set
X = one_hot_encoded_data.drop(['Personnel_needed'], axis=1)
Y = one_hot_encoded_data['Personnel_needed']

#print('test', X.describe())

# use 60% of data for training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

#print('Training data: \n', x_train)

# run a linear regression model using above data
clf = LinearRegression()
clf.fit(x_train, y_train)

#print('intercept of linear regression: ', clf.intercept_)
#print('coefficient of linear regression: ', clf.coef_)

y_pred = clf.predict(x_test)
# example model predictions vs actual observed values
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2)



print('\n PREDICTED VALUES:')
print(clf.predict(x_test))
print('\n ACCURACY SCORE OF MODEL: ', clf.score(x_test, y_test))


print('\n* MODEL STATS * \n')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



#TESTS
# print(df)
# df = df.groupby('Dish')


# print(df)
# print('here')
# print(df['Personnel_needed'].describe(include='all'))
# print(df['Preparation_duration'].describe(include='all'))
# print(df['Consumption_duration'].describe(include='all'))
# print(df['Serving_duration'].describe(include='all'))
# print(df['Spice_density'].describe(include='all'))
