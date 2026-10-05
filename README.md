# Programando un red neuronal

Las redes neuronales están compuestas por neuronas, que a su vez se agrupan en capas.
Cada neurona de cada capa se conecta con todas las neuoronas de la capala anterior y en cada una se realizan las siguientes operaciones:

- _La capa recibe inputs_: La primera recibe datos de entrada y las siguientes capas reciben el input de la capa anterior.
- _Se realiza una suma ponderada de todos los valores de entrada_: matriz W donde las filas son las neuronas de la capa anterior y las columnas las neuronas de la capa actual.
- _Al resultado se le suma otro parámetro llamado bias(b)_
- _Se le aplica una función de activación_: Esto es porque basicamente estamos realizando una regresión lineal, es decir, devuelve un valor lineal.
