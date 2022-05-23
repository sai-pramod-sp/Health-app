import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Label encoding for handling categorical values
le=LabelEncoder()
for i in list(df.columns):
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])

# Dropping the unnamed column
df = df.drop("Unnamed: 32", axis=1)

# Reading the csv file
data=pd.read_csv("cancer.csv")

# Dropping the un used columns
data.drop(["Unnamed: 32"],axis="columns",inplace=True)
data.drop(["id"],axis="columns",inplace=True)

# Data Preprocessing
a=pd.get_dummies(data["diagnosis"])
cancer=pd.concat([data,a],axis="columns")
cancer.drop(["diagnosis","B"],axis="columns",inplace=True)
cancer.rename(columns={"M":"Malignant/Benign"},inplace=True)
y=cancer[["Malignant/Benign"]]
X=cancer.drop(["Malignant/Benign"],axis="columns")
print(X.shape[1]) 


# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Normalization
scaler = StandardScaler()
x = scaler.fit_transform(X)


X=np.array(X)
y=np.array(y)

##logreg.fit(X,y.reshape(-1,))

#joblib.dump(logreg,"model")

rand_clf = RandomForestClassifier(criterion = 'gini', max_depth = 3, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 4, n_estimators = 180)
rand_clf.fit(X, y.reshape(-1,))


joblib.dump(rand_clf,"model")




