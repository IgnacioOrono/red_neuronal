import numpy as np

# el MSE basicamente resta el valor real al esperado y lo eleva al cuadrado. Aprovechamos también para que nos devuelva la derivada de la función coste.
def mse(Yesperada,Yreal):

   #Calculamos error
   x = (np.array(Yesperada) - np.array(Yreal)) ** 2
   x = np.mean(x) #calculo promedio

   #Calulamos la derivada de la función
   y = np.array(Yesperada) - np.array(Yreal)
   return(x,y)