# Se utilizarán ambas funciones.
# La función sigmoide nos va a servir para la ultima capa ya que nos permitira decidir la probabilidad de un estado.
# La función relu es mas sencilla y se aplicará en todas las capas anteriores.

import numpy as np
import math
import matplotlib.pyplot as plt

sigmoid = (
    lambda x: 1 / (1 + np.exp(-x)),
    lambda x: x * (1 - x)
)

def derivada_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


relu = (
    lambda x: x * (x>0),
    lambda x: derivada_relu(x)
)
