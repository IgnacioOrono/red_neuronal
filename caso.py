from scipy import stats
import matplotlib.pyplot as plt
import math
import numpy as np
#--
from capa import *
from entrenamiento import *
from f_activacion import *

# Caso práctico de machine learning
# Problema: clasificación de puntos de dos nubes de puntos

#Vamos a crear primero el contexto.
#Creamos una función que nos devuelva puntos aleatorios alrededor de un circulo de radio R

def circulo(num_datos = 100,R = 1, minimo = 0,maximo= 1):
  pi = math.pi
  r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size= num_datos)) * 10
  theta = stats.truncnorm.rvs(minimo, maximo, size= num_datos) * 2 * pi *10

  x = np.cos(theta) * r
  y = np.sin(theta) * r

  y = y.reshape((num_datos,1))
  x = x.reshape((num_datos,1))

  #Reducimos el numero de elementos para que no cause un Overflow
  x = np.round(x,3)
  y = np.round(y,3)

  df = np.column_stack([x,y])
  return(df)


#Ahora creamos dos sets de datos aleatorios, estos puntos pueden solaparse de tal forma que le cueste más aprender.
datos_1 = circulo(num_datos = 150, R = 2)
datos_2 = circulo(num_datos = 150, R = 0.5)

X = np.concatenate([datos_1,datos_2])
X = np.round(X,3)

Y = [0] * 150 + [1] * 150
Y = np.array(Y).reshape(len(Y),1)


neuronas = [2,4,8,1] 
funciones_activacion = [relu,relu, sigmoid]
red_neuronal = []

for paso in list(range(len(neuronas)-1)):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])
  red_neuronal.append(x) 

error = []
predicciones = []

#epoch sería pasada completa
for epoch in range(0,1000):
  ronda = entrenamiento(X = X ,Y = Y ,red_neuronal = red_neuronal, lr = 0.001)
  predicciones.append(ronda)
  temp = mse(np.round(predicciones[-1]),Y)[0]
  error.append(temp)

epoch = list(range(0,1000))
plt.plot(epoch, error)
plt.show()