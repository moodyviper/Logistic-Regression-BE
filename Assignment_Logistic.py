#-------------Logistic Regression------------------------------
#Import Libraries
import pandas as pd
import seaborn as sns

#Import data 
HR_data = pd.read_csv("D:/Machine_Learning/AE_Files/T7_HR_Data.csv")
HR_data
HR_data.head(5)
HR_data.tail(5)

print("No. of passengers in original dataset:" +str(len(HR_data.index)))

#Analyzing Data
sns.countplot(x="survived",data=HR_data)

sns.countplot(x="survived",hue="sex",data=HR_data)

sns.countplot(x="survived",hue="pclass",HR_data)

#CHECKING DATA TYPE OF A VARIABLE AND CONVERTING IT INTO ANOTHER TYPE-----
HR_data.info()
HR_data["age"].plot.hist()

#Converting var "age" from object type to float type
HR_data["age"] = pd.to_numeric(HR_data.age, errors='coerce')
HR_data.info()
#Parameter: errors = 'coerce' in above fxn, replaces missing values (like "?") if any
#in "age" column by "nan" values.

HR_data["age"].plot.hist()

#Converting var "fare" from object type to float type
HR_data["fare"] = pd.to_numeric(HR_data.fare, errors='coerce')
HR_data.info()
#Parameter: errors = 'coerce' in above fxn, replaces missing values (like "?") if any
#in "fare" column by "nan" values.

HR_data["fare"].plot.hist()

#Identifying/Finding missing values if any----
HR_data.isnull()
HR_data.isnull().sum()

sns.heatmap(HR_data.isnull(),yticklabels=False, cmap="viridis")

#Note: 
#Since missing values in "fare" are quite less, we can delete such rows.
#Since missing values in "age" are high, its better we do imputation in it.

sns.boxplot(x="age",data=HR_data)
sns.boxplot(x="fare",data=HR_data)

#By boxplot we observe that the no. of outliers in "age" are quite less, hence,
#if we plan to do imputation in "age" we can do it by "mean" imputation.

#Handling Missing Values------------
HR_data.head(5)

#Droping all the rows which have a missing value in column (Fare)
#Drop NaN in a specific column
HR_data.dropna(subset=['fare'],inplace=True)
sns.heatmap(HR_data.isnull(),yticklabels=False)

#Imputing missing values in column (Age) with mean imputation
HR_data["age"].fillna(HR_data["age"].mean(), inplace=True)
sns.heatmap(HR_data.isnull(),yticklabels=False)

#Hence, we do not have any missing values in the dataset now.
HR_data.isnull().sum()

#Note:
#A Heat map is usually drawn for either continuous of categorical var
#Lets take few cont var columns and draw the heat map
#Cont = HR_data[:,[5,6,7]]
#sns.heatmap(Cont)

#There are lot of string value var in dataset which have to be converted to numerical
#values for applying machine learing algoritm. Hence, we will now convert string var 
#to numerical var.
HR_data.info()
pd.get_dummies(HR_data["sex"])

pd.get_dummies(HR_data["sex"],drop_first=True)

Sex_Dummy = pd.get_dummies(HR_data["sex"],drop_first=True)
Sex_Dummy.head(5)

pd.get_dummies(HR_data["embarked"])
Embardked_Dummy = pd.get_dummies(HR_data["embarked"],drop_first=True)
Embardked_Dummy.head(5)

pd.get_dummies(HR_data["pclass"])
PClass_Dummy = pd.get_dummies(HR_data["pclass"],drop_first=True)
PClass_Dummy.head(5)

#Now, lets concatenate these dummy var columns in our dataset.
HR_data = pd.concat([HR_data,Sex_Dummy,PClass_Dummy,Embardked_Dummy],axis=1)
HR_data.head(5)

#dropping the columns whose dummy var have been created
HR_data.drop(["sex","embarked","pclass","Passenger_id","name","ticket"],axis=1,inplace=True)
HR_data.head(5)

#Splitting the dataset into Train & Test dataset
x=HR_data.drop("survived",axis=1)
y=HR_data["survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

#Hence, accuracy = (165+84)\(165+84+30+44) = 77.5%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)

#----To Improve the accuracy of the model, lets go with Backward ELimination Method &
# rebuild the logisitc model again with few independent variables--------
HR_data_1 = HR_data
HR_data_1.head(5)

#--------------------------Backward Elimination--------------------------------
#Backward elimination is a feature selection technique while building a machine learning model. It is used
#to remove those features that do not have significant effect on dependent variable or prediction of output.

#Step: 1- Preation of Backward Elimination:
#Importing the library:
import statsmodels.api as sm

#Adding a column in matrix of features:
x1=HR_data_1.drop("survived",axis=1)
y1=HR_data_1["survived"]
import numpy as nm
x1 = nm.append(arr = nm.ones((1291,1)).astype(int), values=x1, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10]]

#for fitting the model, we will create a regressor_OLS object of new class OLS of statsmodels library. 
#Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()

#In the above summary table, we can clearly see the p-values of all the variables. 
#And remove the ind var with p-value greater than 0.05
x_opt= x1[:, [0,1,2,4,5,6,7,8,9,10]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,4,5,6,7,9,10]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,5,6,7,9,10]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,5,6,7,10]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()
#Hence,independent var - age, sibsp, sex, pclass & embarked are significant variable 
#for the predicting the value of Dependent Var "survived".
#So we can now predict efficiently using these variables.

#-------Building Logistic Regression model using ind var: age, sibsip, sex, pclass & embarked--------  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.25, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,predictions)

#Accuracy = (170+87)/(170+87+25+41) = 80%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)
