from sklearn.datasets import load_breast_cancer
from RegresionLogistica import LogisticRegression
import pandas as pd

#Cargar data
data = pd.read_csv("Grupo8Preprocesamiento.csv", index_col=0)
data=pd.get_dummies(data,columns=['ES INGRESANTE?'], drop_first=True)
 
#Para dividir conjunto de datos entrenamiento y pruebas
from sklearn.model_selection import train_test_split


X_df = data.drop('ES INGRESANTE?_SI',axis=1)
X = X_df.to_numpy()

y_df = data['ES INGRESANTE?_SI']
y = y_df.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=100)

regressor = LogisticRegression(X_train,y_train)

regressor.fit(0.1 , 5000)

y_pred = regressor.predict(X_train,0.5)

print('accuracy -> {}'.format(sum(y_pred == y_train) / y_train.shape[0]))