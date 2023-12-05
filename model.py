import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

df= pd.read_csv('diabetes_prediction_dataset.csv')

#data preprocessing
df['age'] = df['age'].astype(int)
df = df[df['gender']!='Other']  
df = df.drop_duplicates()
df['gender']=df['gender'].map({'Female':1,'Male':0})
df['smoking_history']=df['smoking_history'].map({'No Info':0,'never':1,'former':2,'current':3,'not current':4,'ever':5})
df = df[df['diabetes']!='nan']

X = df.drop(columns = ["diabetes"])
Y = df["diabetes"]

smote=SMOTE(sampling_strategy='minority')
X,Y=smote.fit_resample(X,Y)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state = 0)

# Normalization(scaling) 
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

#Run a classifier
model= RandomForestClassifier(n_estimators= 10, criterion="entropy")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


print(classification_report(y_test,y_pred))

pickle.dump(model, open('model1.pkl', 'wb'))
model1=pickle.load(open('model1.pkl','rb'))
