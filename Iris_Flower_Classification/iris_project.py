import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

Iris_data = pd.read_csv('Iris.csv') 

# print(Iris_data.head()) 

# print(Iris_data.info())

# print(Iris_data.describe())

Iris_data = Iris_data.drop('Id',axis=1)

le = LabelEncoder()
Iris_data['Species_encoded'] = le.fit_transform(Iris_data['Species'])
# print(le.classes_)

# print(Iris_data[['Species','Species_encoded']].head(10))

X =Iris_data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= Iris_data['Species_encoded']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("Accuracy: " , accuracy_score(y_test,y_predict))

print("\n Model Performance Report:\n")
print(classification_report(y_test, y_predict, target_names=le.classes_))

cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Prediction
new_flower= pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
pred = model.predict(new_flower)
print("Predicted Species:", le.inverse_transform(pred))


