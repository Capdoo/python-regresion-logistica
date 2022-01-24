from RegresionLogistica import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def gradients(X, y, y_hat):
    
    # X --> Entrada.
    # y --> true/target valores.
    # y_hat --> hipotesis/predicciones.
    # w --> pesos (parametros).
    # b --> bias (parametros).
    
    # m-> numero de ejmplos de entrenamiento.
    m = X.shape[0]
    
    # Gradiente de perdida w.r.t wesos.
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    
    # Gradiente de perdida w.r.t bias.
    db = (1/m)*np.sum((y_hat - y)) 
    
    return dw, db



def normalize(X):
    
    # X --> Entrada.
    
    # m-> numero de ejemplos de entrenamient0
    # n-> numero de caracteristicas 
    m, n = X.shape
    
    # Normalizando todas las n caracteristicas de X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        
    return X




def train(X, y, bs, epochs, lr):
    
    # X --> Entrada.
    # y --> true/target valor.
    # bs --> Tamaño del batch o lote.
    # epochs --> Numero de iteraciones.
    # lr --> Tasa de aprendizaje.
        
    # m-> numero de ejemplos de entrenamiento
    # n-> numero de caracteristicas o campos
    m, n = X.shape
    
    # Inicializando pesos y bias a cero.
    w = np.zeros((n,1))
    b = 0
    
    # Reshape de y.
    y = y.reshape(m,1)
    
    # Normalizando las entradas.
    x = normalize(X)
    
    # Lista vacia para almancenar las perdidas
    losses = []
    
    # Bucle de entrenamiento.
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            
            # Definiendo lotes. SGD.
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculando hipotesis/prediccion.
            y_hat = sigmoid(np.dot(xb, w) + b)
            
            # Obteniendo la gradiente de perdida, parametros w.r.t.
            dw, db = gradients(xb, yb, y_hat)
            
            # Actualizando los parametros.
            w -= lr*dw
            b -= lr*db
        
        # Calculando la perdida y append en la lista.
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
        
    # retornando pesos, bias y perdidas(Lista).
    return w, b, losses

def predict(X):
    
    # X --> Entrada.
    
    # Normalizando las entradas.
    x = normalize(X)
    
    # Calculando predicciones/y_hat.
    preds = sigmoid(np.dot(X, w) + b)
    
    # Lista vacia para almacenar las predicciones.
    pred_class = []
    # if y_hat >= 0.5 --> redondear a 1
    # if y_hat < 0.5 --> redondear a 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    
    return np.array(pred_class)

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy


data = pd.read_csv("Grupo8Preprocesamiento.csv", index_col=0)
data=pd.get_dummies(data,columns=['ES INGRESANTE?'], drop_first=True)

#Para dividir conjunto de datos entrenamiento y pruebas
from sklearn.model_selection import train_test_split


X_df = data.drop('ES INGRESANTE?_SI',axis=1)
X = X_df.to_numpy()

y_df = data['ES INGRESANTE?_SI']
y = y_df.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=100)

# Entrenando 
w, b, l = train(X_train, y_train, bs=100, epochs=1000, lr=0.01)

accuracy(y_test, predict(X_test))


prediccion_test = predict(X_test)
print(prediccion_test)



alumno_prueba = data.sample()

alumno_x = alumno_prueba.drop('ES INGRESANTE?_SI',axis=1)
alumno_y = alumno_prueba['ES INGRESANTE?_SI']

a = alumno_x.to_numpy()
print("Este es el alumno",a)


prediccion_final = predict(a)
print("Esta es la prediccion",prediccion_final)

print("Esto deberia salir:", alumno_y.to_numpy())

