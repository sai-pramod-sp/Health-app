
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.externals import joblib
df=pd.read_csv("diabetes.csv")

#replacing 0 values with Nan
df['SkinThickness'].replace(0, np.nan,inplace=True)
df['BloodPressure'].replace(0, np.nan,inplace=True)
df['Insulin'].replace(0, np.nan,inplace=True)
df['Glucose'].replace(0, np.nan,inplace=True)
df['BMI'].replace(0, np.nan,inplace=True)

#fill misssing vlaues according to outcome
for col in df.columns: 
    df.loc[(df["Outcome"]==0) & (df[col].isnull()),col] = df[df["Outcome"]==0][col].median()
    df.loc[(df["Outcome"]==1) & (df[col].isnull()),col] = df[df["Outcome"]==1][col].median()

#change numeric data to categorical
df['Glucose'] = pd.cut(x=df['Glucose'], bins=[0,139,200,1000],labels = [1,2,3]) 
df['BMI'] = pd.cut(x=df['BMI'], bins=[0,18.5,24.9,29.9,100],labels = [1,2,3,4])
df['BloodPressure'] = pd.cut(x=df['BloodPressure'], bins=[0,79,89,119,500],labels = [1,2,3,4])

X=df.drop("Outcome",axis=1) #dropping target value
y=df[["Outcome"]] #keeping target value

scaler=StandardScaler() #scale data
X = scaler.fit_transform(X)

# Training the model using Random forest Classifier
randomforest_classifier=RandomForestClassifier()

param_grid = {
    'max_depth': [10, 50,100],
    'n_estimators': [200, 400]}
clf_lr = GridSearchCV(randomforest_classifier, param_grid = param_grid, cv = 5)


clr_lr.fit(X,y)


'''# DATA FOR PRED
data=pd.read_csv("diabetes.csv")
print(data.head())


logreg=LogisticRegression()



X=data.iloc[:,:8]
print(X.shape[1])


y=data[["Outcome"]]

X=np.array(X)
y=np.array(y)

logreg.fit(X,y.reshape(-1,))'''
joblib.dump(clr_lr,"model1")