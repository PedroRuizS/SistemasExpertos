import numpy as np
from sklearn.model_selection import train_test_split
from datos_album import album_valor, album_nombre
from algo import KNN
import random

albumRandom = random.choice(album_nombre)
X, y = album_valor, album_nombre

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#definimos q queremos 3 recomendaciones
clf = KNN(k=3)
clf.fit(X_train, y_train)

####################################
####################################
####################################

#input del usuaripo
print("Ingrese su calificación del album:")
print(albumRandom)
valor = float(input(" "))

####################################
####################################
#usando el valor, buscamos los vecinos mas cercanos
distanciasK, indices = clf.kVecinos([valor])

print("Las recomendaciones son:")
for i in indices[0]:
    print(f"{X_train[i]} de calificación para el album {y_train[i]}")
