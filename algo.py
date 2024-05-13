import numpy as np
from collections import Counter

##https://www.youtube.com/watch?v=rTEtEy5o3X0&t=253s


def distanciaEuclid(x1, x2):
    distancia = np.sqrt(np.sum((x1-x2)**2))
    return distancia

class KNN:
    def __init__(self, k=3):
        self.k = k

    def kVecinos(self, X):
        distanciasK = [self.calcularDistancia(x) for x in X]
        indices = np.argsort(distanciasK)[:, :self.k]
        return distanciasK, indices

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def calcularDistancia(self, x):
        distances = [distanciaEuclid(x, x_train) for x_train in self.X_train]
        return distances

    def prediccionCompleta(self, X):
        predictions = [self._prediccion(x) for x in X]
        return predictions

    def _prediccion(self, x):
        distancias = [distanciaEuclid(x, x_train) for x_train in self.X_train]
    
        kIndices = np.argsort(distancias)[:self.k]
        kLabelsCercanos = [self.y_train[i] for i in kIndices]

        masComun = Counter(kLabelsCercanos).masComun()
        return masComun[0][0]