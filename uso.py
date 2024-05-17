import numpy as np
from sklearn.model_selection import train_test_split
from datos_album import album_valor, album_nombre, album_genero, album_resena  # Agrega album_resena
from algo import KNN
import tkinter as tk
from tkinter import messagebox
import random



albumRandom = random.choice(album_nombre)
X, y, z = album_valor, album_nombre, album_genero  # Agregamos z para los géneros

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = KNN(k=3)
clf.fit(X_train, y_train, z)  # Añadimos z como el tercer argumento


####################################
####################################
####################################
def enviar_calificacion():
    # Obtenemos el valor ingresado por el usuario
    valor = float(entry_calificacion.get())

    # Obtenemos las recomendaciones basadas en el valor ingresado
    _, indices = clf.kVecinos([valor])

    # Mostramos las recomendaciones en una ventana emergente
    recomendaciones = ""
    for i in indices[0]:
        recomendaciones += f"Álbum: {album_nombre[i]}\n"
        recomendaciones += f"Calificación: {album_valor[i]}\n"
        recomendaciones += f"Reseña: {album_resena[i]}\n\n"  # Agregar la reseña
    messagebox.showinfo("Recomendaciones", recomendaciones)


def mostrar_recomendacion_aleatoria():
    # Seleccionar aleatoriamente un álbum y su calificación
    indice_aleatorio = random.randint(0, len(album_nombre) - 1)
    album_aleatorio = album_nombre[indice_aleatorio]
    calificacion_aleatoria = album_valor[indice_aleatorio]
    reseña_aleatoria = album_resena[indice_aleatorio]  # Agregar la reseña aleatoria

    # Mostrar la recomendación en una ventana emergente
    messagebox.showinfo("Recomendación Aleatoria", f"Álbum: {album_aleatorio}\nCalificación: {calificacion_aleatoria}\nReseña: {reseña_aleatoria}")


###############################################################################
###############################################################################
###############################################################################

ventana = tk.Tk()
ventana.title("Sistema de Recomendación de Música")
ventana.geometry("400x300")  # Tamaño de la ventana

# Creamos las etiquetas y campos de texto
etiqueta_calificacion = tk.Label(ventana, text="Ingrese su calificación de álbum:", font=("Arial", 10))
etiqueta_calificacion.pack()

etiqueta_album = tk.Label(ventana, text=f"{albumRandom}", font=("Arial", 12))
etiqueta_album.pack()


entry_calificacion = tk.Entry(ventana, font=("Arial", 10))
entry_calificacion.pack()

# Creamos el botón "Enviar"
boton_enviar = tk.Button(ventana, text="Enviar", font=("Arial", 10), command=enviar_calificacion)
boton_enviar.pack()

boton_aleatorio = tk.Button(ventana, text="Recomendación Aleatoria", font=("Arial", 10), command=mostrar_recomendacion_aleatoria)
boton_aleatorio.pack()


#input del usuario

print("Ingrese su calificación del álbum:")
print(albumRandom)
valor = float(input(" "))


####################################
####################################
#usando el valor, buscamos los vecinos más cercanos
distanciasK, indices = clf.kVecinos([valor])


print("Las recomendaciones son:")
for i in indices[0]:
    print(f"{X_train[i]} de calificación para el álbum {y_train[i]}")
