import numpy as np
from collections import Counter

# Function to calculate Euclidean distance between two points
def distanciaEuclid(x1, x2):
    # Calculate the square of the differences
    diff_sqr = (x1 - x2)**2
    # Sum the squared differences
    sum_diff_sqr = np.sum(diff_sqr)
    # Calculate the square root of the sum
    distancia = np.sqrt(sum_diff_sqr)
    return distancia

# K-Nearest Neighbors (KNN) class
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []
        self.z_train = []  # Lista para almacenar los géneros

    def fit(self, X, y, z):  # Añadimos z para los géneros
        self.X_train = X
        self.y_train = y
        self.z_train = z  # Almacenamos los géneros

    def kVecinos(self, X):
        distanciasK = [self.calcularDistancia(x) for x in X]
        indices = np.argsort(distanciasK)[:, :self.k]
        return distanciasK, indices

    def calcularDistancia(self, x):
        distances = [distanciaEuclid(x, x_train) for x_train in self.X_train]
        return distances

    def prediccionCompleta(self, X):
        predictions = [self._prediccion(x) for x in X]
        return predictions

    def _prediccion(self, x):
        distancias = [distanciaEuclid(x, x_train) for x_train in self.X_train]

        # Tomamos los indices de los vecinos mas cercanos
        kIndices = np.argsort(distancias)[:self.k]

        # Tomamos los géneros de los vecinos mas cercanos
        kGenerosCercanos = [self.z_train[i] for i in kIndices]

        # Tomamos el género más común entre los vecinos más cercanos
        genero_predicho = Counter(kGenerosCercanos).most_common(1)[0][0]

        return genero_predicho
