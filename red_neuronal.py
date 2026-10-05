from capa import capa
from f_activacion import relu,sigmoid
import numpy as np

# Neuronas en cada capa, el primer valor es el numero de columnas de la capa de entrada (dos variables)
# Dos capas ocultas de 4 y 8
# Y la ultima una capa de salida que predecirá la clase
neuronas = [2,4,8,1]

#Funciones de activación que vamos a usar en cada capa.
funciones_activacion = [relu, relu, sigmoid]

#objeto red_neuronal
red_neuronal = []

# Estructura de una red neuronal
for paso in range(len(neuronas) - 1):
    x = capa(neuronas[paso], neuronas[paso+1],funciones_activacion[paso])
    red_neuronal.append(x)


#Predicción de una red neuronal

X = np.round(np.random.randn(20,2),3) #ejemplo de vector de entrada

output = [X]

for num_capa in range(len(red_neuronal)):
  z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b # multiplicación matricial y suma del bias
  a = red_neuronal[num_capa].funcion_act[0](z) #aplicamos la funcion de activación
  output.append(a)

print(f'Primera ronda, la red sin entrenamiento \n {output[-1]}')


# Entrenamiento de la red neuronal
# para entrenar una red neuronal tenemos que saber cuanto falló: usamos el MSE(error cuadrático medio) esto seria nuestra funcion de coste.
# el MSE basicamente resta el valor real al esperado y lo eleva al cuadrado. Aprovechamos también para que nos devuelva la derivada de la función coste.
def mse(Yesperada,Yreal):

   #Calculamos error
   x = (np.array(Yesperada) - np.array(Yreal)) ** 2
   x = np.mean(x) #calculo promedio

   #Calulamos la derivada de la función
   y = np.array(Yesperada) - np.array(Yreal)
   return(x,y)




