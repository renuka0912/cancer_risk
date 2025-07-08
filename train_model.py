import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('patient.csv')
df['gender']=LabelEncoder().fit_transform(df['gender'])
le_breast=LabelEncoder()
le_cervical=LabelEncoder()
le_colorectal=LabelEncoder()
df['risk_breast_encoded']=le_breast.fit_transform(df['risk_breast'])
df['risk_cervical_encoded']= le_cervical.fit_transform(df['risk_cervical'])
df['risk_colorectal_encoded'] =le_colorectal.fit_transform(df['risk_colorectal'])
df['has_family_history']=df['family_history'].apply(lambda x: 0 if x.strip().lower() == 'none' else 1)
df['is_smoker']=df['lifestyle'].apply(lambda x:1 if'smoker'in x.lower()else 0)
df['is_obese']=df['lifestyle'].apply(lambda x:1 if'obese'in x.lower() else 0)
df['uses_alcohol'] df['lifestyle'].apply(lambda x:1 if'alcohol'in x.lower()else 0)
features = ['age','gender','has_family_history','is_smoker','is_obese','uses_alcohol']
model_breast=RandomForestClassifier(random_state=42).fit(df[features], df['risk_breast_encoded'])
model_cervical =RandomForestClassifier(random_state=42).fit(df[features], df['risk_cervical_encoded'])
model_colorectal=RandomForestClassifier(random_state=42).fit(df[features], df['risk_colorectal_encoded'])

joblib.dump(model_breast,'breast_model.pkl')
joblib.dump(model_cervical,'cervical_model.pkl')
joblib.dump(model_colorectal,'colorectal_model.pkl')
joblib.dump(le_breast,'le_breast.pkl')
joblib.dump(le_cervical,'le_cervical.pkl')
joblib.dump(le_colorectal,'le_colorectal.pkl')
print("Models and encoders saved successfully.")
