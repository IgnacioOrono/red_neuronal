from capa import capa
from f_activacion import relu,sigmoid
from mse import mse
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

#print(f'Primera ronda, la red sin entrenamiento \n {output[-1]}')


# Entrenamiento de la red neuronal
# para entrenar una red neuronal tenemos que saber cuanto falló: usamos el MSE(error cuadrático medio) esto seria nuestra funcion de coste.

from random import shuffle

Y = [0] * 10 + [1] * 10 
shuffle(Y)
Y = np.array(Y).reshape(len(Y),1)

print (mse(output[-1], Y)[0])

# Backpropagation y gradient descent

# El algoritmo de gradient descent basicamente utiliza el inverso del gradiente (derivadas parciales) para saber que valor es el óptimo.
# Otro parámetro  que se aplica a este sería el "learning rate" que sería la cantidad de pasos, es decir, si es un valor bajo vamos a tardar más
# porque cada paso será muy corto. Si es muy grande puede que nos pasemo del valor óptimo y nunca lo encontremos.
# Para este caso lo vamos a dejar fijo, pero en realidad lo óptimo sería ir disminuyendo el learning rate en cada paso que demos.

# Con Backpropagation(backprop) vamos a transmitir el error hacia atrás por lo que se inicia calculando el error en la última capa.

# Definimos el learning rate
lr = 0.05

# Creamos el indice inverso para ir de derecha a izquierda
back = list(range(len(output)-1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa
delta = []

for capa in back:
  # Backprop
  # Guardamos los resultados de la ultima capa antes de usar backprop para poder usarlas en gradient descent
  a = output[capa+1][1]

  # Backprop en la ultima capa 
  if capa == back[0]:
    x = mse(a,Y)[1] * red_neuronal[capa].funcion_act[1](a)
    delta.append(x)

  # Backprop en el resto de capas 
  else:
    x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
    delta.append(x)

  # Guardamos los valores de W para poder usarlos en la iteracion siguiente
  W_temp = red_neuronal[capa].W.transpose()

  # Gradient Descent #
  # Ajustamos los valores de los parametros de la capa
  red_neuronal[capa].b = red_neuronal[capa].b - delta[-1].mean() * lr
  red_neuronal[capa].W = red_neuronal[capa].W - (output[capa].T @ delta[-1]) * lr


#print('MSE: ' + str(mse(output[-1],Y)[0]) )
#print('Estimacion: ' + str(output[-1]) )




