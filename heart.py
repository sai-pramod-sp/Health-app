import pandas as pd
import numpy as np
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# from sklearn.linear_model import LogisticRegression
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("heart.csv")

df = df[df['ca'] < 4] #drop the wrong ca values
df = df[df['thal'] > 0] # drop the wong thal value

df = df.rename(columns = {'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure', 'chol': 'cholesterol','fbs': 'fasting_blood_sugar', 
                       'restecg' : 'rest_electrocardiographic', 'thalach': 'max_heart_rate_achieved', 'exang': 'exercise_induced_angina',
                       'oldpeak': 'st_depression', 'slope': 'st_slope', 'ca':'num_major_vessels', 'thal': 'thalassemia'}, errors="raise")

# numerical fearures 6
num_feats = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
# categorical (binary)
bin_feats = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'target']
# caterorical (multi-)
nom_feats= ['chest_pain_type', 'rest_electrocardiographic', 'st_slope', 'thalassemia']
cat_feats = nom_feats + bin_feats

def label_encode_cat_features(data, cat_features):
    '''
    Given a dataframe and its categorical features, this function returns label-encoded dataframe
    '''
    
    label_encoder = LabelEncoder()
    data_encoded = data.copy()
    
    for col in cat_features:
        data_encoded[col] = label_encoder.fit_transform(data[col])
    
    data = data_encoded
    
    return data

cat_features = cat_feats
df = label_encode_cat_features(df, cat_features)
seed = 0

x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

rand_clf = RandomForestClassifier(max_depth= 70,
               max_features='auto',
               min_samples_leaf= 4,
               min_samples_split= 10,
               n_estimators= 400,random_state=seed)


rand_clf.fit(x, y)
joblib.dump(rand_clf,"model2")

