import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# open and read the csv file
df = pd.read_csv('menu_info.csv')
# print head of DataFrame
print(df.head())
# define the columns
columns = [ 'Preparation_duration', 'Waiting_duration', 'Price', 'Consumption_duration',
           'Serving_duration', 'Spice_density', 'Personnel_needed']
print(df.head())



# provide basic statistics for data
print(df.describe())
# info on types of vars
print(df.info())

#sns.pairplot(df[columns], size=2.0)
#plt.show()

stdsc = StandardScaler()
X_std = stdsc.fit_transform(df[columns].iloc[:,range(0,7)].values)

cov_mat =np.cov(X_std.T)

#plt.figure(figsize=(10,10))
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cov_mat,
#                 cbar=True,
#                 annot=True,
#                 square=True,
#                 fmt='.2f',
#                 annot_kws={'size': 12},
#                 yticklabels=columns,
#                 xticklabels=columns)
#plt.title('Covariance matrix showing correlation coefficients')
#plt.tight_layout()
#plt.show()



# check covariance among variables
#corr1 = one_hot_encoded_data.corr()
#print(corr1[corr1 < 1].unstack().transpose() \
#      .sort_values(ascending=False) \
#      .drop_duplicates())

select_cols = ['Waiting_duration', 'Price', 'Serving_duration', 'Consumption_duration', 'Personnel_needed']
print('HERE \n',df[select_cols].head())


X = df[select_cols].iloc[:,0:4].values
y = df[select_cols]['Personnel_needed'].values

print(X.shape)
print(y.shape)

# one-hot encoding the categorical variables
#ohe = OneHotEncoder(categorical_features=[0 ,1])
one_hot_encoded_data = pd.get_dummies(df, columns=['Dish_name', 'Restaurant'])
#print(one_hot_encoded_data)

#print('test', X.describe())

X = df[select_cols].iloc[:,0:4].values
y = df[select_cols]['Personnel_needed']


# use 60% of data for training
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0)

# used for SVM check
#x_train = x_train.astype(int)
#x_test = x_test.astype(int)
#y_train = y_train.astype(int)
#y_test = y_test.astype(int)

#print('Training data: \n', x_train)

# run a linear regression model using above data

clf = LinearRegression()
#clf = svm.SVC(kernel='linear')   <- TEST
clf.fit(x_train, y_train)

#print('intercept of linear regression: ', clf.intercept_)
#print('coefficient of linear regression: ', clf.coef_)

# run regression model predict data
y_test_pred = clf.predict(x_test)
y_train_pred = clf.predict(x_train)
# example model predictions vs actual observed values
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
print(df2)

# BUILDING THE MULTI REGRESSION MODEL BELOW
#
# scatter plot training vs testing predicted data coeff.
plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.legend(loc='lower right')
plt.show()

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


#clf.fit(x_train, y_train).intercept_
#clf.fit(x_train, y_train).coef_

#
# FEATURES STANDARDIZATION, HYPER-PARAM TUNING, X VALIDATION
#

sc_y = StandardScaler()
sc_x = StandardScaler()
y_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()

train_score = []
test_score = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=i)
    y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=4)),('slr', LinearRegression())])
    pipe_lr.fit(X_train, y_train_std)
    y_train_pred_std=pipe_lr.predict(X_train)
    y_test_pred_std=pipe_lr.predict(X_test)
    y_train_pred=sc_y.inverse_transform(y_train_pred_std)
    y_test_pred=sc_y.inverse_transform(y_test_pred_std)
    train_score = np.append(train_score, r2_score(y_train, y_train_pred))
    test_score = np.append(test_score, r2_score(y_test, y_test_pred))

# evaluation of regression model stats
print('R2 train: %.3f +/- %.3f' % (np.mean(train_score),np.std(train_score)))
print('R2 test: %.3f +/- %.3f' % (np.mean(test_score),np.std(test_score)))




#print('\n PREDICTED VALUES:')
#print(clf.predict(x_test))
print('\n ACCURACY SCORE OF MODEL: ', clf.score(x_test, y_test))

#print('\n* MODEL STATS * \n')
#print('Pearson Correlation:\n', df2.corr(method = 'pearson'))
#print("Precision:", metrics.precision_score(y_test, y_pred))
#print('Pearson Correlation on training set:', pearsonr(x_train.values(), y_train.values()))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

#
# PCA - PRINCIPAL COMPONENT ANALYSIS
#

train_score = []
test_score = []
cum_variance = []

for i in range(1,5):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)
    y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=i)),('slr', LinearRegression())])
    pipe_lr.fit(X_train, y_train_std)
    y_train_pred_std=pipe_lr.predict(X_train)
    y_test_pred_std=pipe_lr.predict(X_test)
    y_train_pred=sc_y.inverse_transform(y_train_pred_std)
    y_test_pred=sc_y.inverse_transform(y_test_pred_std)
    train_score = np.append(train_score, r2_score(y_train, y_train_pred))
    test_score = np.append(test_score, r2_score(y_test, y_test_pred))
    cum_variance = np.append(cum_variance, np.sum(pipe_lr.fit(X_train, y_train).named_steps['pca'].explained_variance_ratio_))


plt.scatter(cum_variance,train_score, label = 'train_score')
plt.plot(cum_variance, train_score)
plt.scatter(cum_variance,test_score, label = 'test_score')
plt.plot(cum_variance, test_score)
plt.xlabel('cumulative variance')
plt.ylabel('R2_score')
plt.legend()
plt.show()


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=0)
y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
X_train_std = sc_x.fit_transform(X_train)
X_test_std = sc_x.transform(X_test)

alpha = np.linspace(0.01,0.4,10)

#
# REGULARIZED LASSO REGRESSION
#

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.7)

r2_train=[]
r2_test=[]
norm = []
for i in range(10):
    lasso = Lasso(alpha=alpha[i])
    lasso.fit(X_train_std,y_train_std)
    y_train_std=lasso.predict(X_train_std)
    y_test_std=lasso.predict(X_test_std)
    r2_train=np.append(r2_train,r2_score(y_train,sc_y.inverse_transform(y_train_std)))
    r2_test=np.append(r2_test,r2_score(y_test,sc_y.inverse_transform(y_test_std)))
    norm= np.append(norm,np.linalg.norm(lasso.coef_))

plt.scatter(alpha,r2_train,label='r2_train')
plt.plot(alpha,r2_train)
plt.scatter(alpha,r2_test,label='r2_test')
plt.plot(alpha,r2_test)
plt.scatter(alpha,norm,label = 'norm')
plt.plot(alpha,norm)
plt.ylim(-0.1,1)
plt.xlim(0,.43)
plt.xlabel('alpha')
plt.ylabel('R2_score')
plt.legend()
plt.show()



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
