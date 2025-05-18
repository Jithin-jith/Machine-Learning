#Import Libraries
import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,f1_score
import seaborn as sns 
import matplotlib.pyplot as plt

data = pd.read_csv("spam.csv")
X = data.drop(columns=['spam'],axis=1)
Y = data['spam']

X_train,X_test,y_train,y_test =  train_test_split(X,Y,test_size=0.2,random_state=42)

#Train the logistic regression model to classify emails as 'spam' or 'not spam'

model = LogisticRegression()
model.fit(X_train,y_train)

#make predictions using the model
y_pred = model.predict(X_test)

#Evaluate the model using accuracy,confusion metrics, precision, recall and F1 score
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1_scores = f1_score(y_test,y_pred)

print(f"Accuracy : {accuracy}")
print(f"precision : {precision}")
print(f"recall : {recall}")
print(f"f1_scores : {f1_scores}")

#visualise the confusion matrics using the seaborn heatmap

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.title('Confusion Metrics')
plt.show()