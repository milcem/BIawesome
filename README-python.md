#         Python task answers and details

Basic statistics of data

          Preparation_duration                 Price           Consumption_duration               
        count            158.000000         158.000000            158.000000        
        mean              15.689873         71.284671             18.457405   
        std                7.615691         37.229540              9.677095   
        min                4.000000         2.329000              0.660000   
        25%               10.000000         46.013000             12.535000   
        50%               14.000000         71.899000             19.500000   
        75%               20.000000         90.772500             24.845000   
        max               48.000000         220.000000             54.000000   

On average, each dish takes **15.68** time units to prepare, costs **71.28** currency units, is consumed in **18.45** time units

          Serving_duration     Waiting_duration    Spice_density   Personnel_needed  
        count        158.000000        158.000000     158.000000        158.000000  
        mean           8.130633          8.830000      39.900949          7.794177  
        std            1.793474          4.471417       8.639217          3.503487  
        min            2.790000          0.330000      17.700000          0.590000  
        25%            7.100000          6.132500      34.570000          5.480000  
        50%            8.555000          9.570000      39.085000          8.150000  
        75%            9.510000         10.885000      44.185000          9.990000  
        max           11.820000         27.000000      71.430000         21.000000  

On average, each dish takes **8.13** time units to be served, waiting duration is **8.83** time units, dishes have an average spice density level of **39.9** and
**7.79** or ~**8** personnel are needed.

![Figure_1](https://user-images.githubusercontent.com/26208356/131999055-13d83ac8-e3a9-4b5a-903d-472218d74744.png)

From the scatter plot above we can visualize the distribution and covariance among the data - e.g. our target Personnel_needed is highly (positively) correlated with Waiting_duration, Price, Consumption_duration, and Serving_duration - which means that if any of those were to increase, how many Personnel are needed would increase as well. We can also tell that the scaling on some of the data varies, e.g. price ranges up to 200 vs the other nominal values range up to 60 - we might need to standardize/normalize this data.

Another way to see highly correlated values in our model is by plotting a covariance matrix heatmap:

![Figure_2 - Covariance matrix](https://user-images.githubusercontent.com/26208356/131999858-713ceaab-b957-48e4-95ef-8c8c21b0922d.png)

From the figure above we can note similarly to the initial scatter plot that the variables that seem to be important when predicting Personnel_needed are 
Waiting_duration, Price, Consumption_duration, and Serving_duration.

We one-hot encode the categorical data - Restaurants and Dish_name in order to be able to run a ML regressor.

We then create a training and testing set where 60% of the data is used for our training model. We run a multi-regression model on our data for predicting "Personnel_needed" in our model - the following scatter plot shows the result of our model:

![Figure_1 - regression model](https://user-images.githubusercontent.com/26208356/132000480-31c4d9eb-1c0a-4b38-823f-77cfe8ac95df.png)

  MSE train: 0.955, test: 0.889
  R^2 train: 0.920, test: 0.928
  
The Mean-Squared-Error of our model on the training data is 0.955 whereas on our test model it is 0.889
R^2 of our training data is 0.920 and on our test it is 0.928 

These values also confirm the positive correlation between our target and independent variable.

Hyper-parameters in this model are our test and training datasets, for which we account by using scikit's feature scaling -  by using scalar.Fit () / flatten() on the Train dataset and let those calculated parameters transform Train and Test data.

Regularization is used to avoid overfitting in our model by reducing the complexity of said model by adding a penalty, complexity, or shrinkage term. 
For this model, I used Lasso regression to show this regularization. In this technique, the L1 penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero which means there is a complete removal of some of the features for model evaluation when the tuning parameter λ is suﬃciently large. Therefore, the lasso method also performs Feature selection and is said to yield sparse models.

![Figure_1 - PCA](https://user-images.githubusercontent.com/26208356/132001820-c5f5407d-7e40-4829-b8c6-6cb5f7edc972.png)

We observe that by increasing the number of principal components from 1 to 4, the train and test scores improve. This is because with less components, there is high bias error in the model, since model is overly simplified. As we increase the number of principal components, the bias error will reduce, but complexity in the model increases.


![Figure_1 - lasso](https://user-images.githubusercontent.com/26208356/132001825-5ff32a53-6dd2-44d5-8051-08d83e96f46f.png)

We observe that as the regularization parameter alpha increases, the norm of the regression coefficients become smaller and smaller. This means more regression coefficients are forced to zero, which intend increases bias error (over simplification). The best value to balance bias-variance tradeoff is when alpha is kept low - alpha =< 0.1 
