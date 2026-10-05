# Programando un red neuronal

Las redes neuronales están compuestas por neuronas, que a su vez se agrupan en capas.
Cada neurona de cada capa se conecta con todas las neuoronas de la capala anterior y en cada una se realizan las siguientes operaciones:

- _La capa recibe inputs_: La primera recibe datos de entrada y las siguientes capas reciben el input de la capa anterior.
- _Se realiza una suma ponderada de todos los valores de entrada_: matriz W donde las filas son las neuronas de la capa anterior y las columnas las neuronas de la capa actual.
- _Al resultado se le suma otro parámetro llamado bias(b)_
- _Se le aplica una función de activación_: Esto es porque basicamente estamos realizando una regresión lineal, es decir, devuelve un valor lineal.

# Problema a resolver

El problema que le pasamos a esta red neuronal es clasificar puntos de dos nubes de puntos, esta sería la forma en la que se ve:

![Contexto](/image/Figure_1.png)

# Resultado

Se obtuvieron las siguientes imágenes que muestran las distintas iteraciones realizadas en las cuales se nota como el eje Y reduce su valor, es decir, el error comienza en un 0.5 y termina en un error de 0.2, y en otro test en menos de 0.1
Esto significa que la red neuronal entrenó y llegó a mejorar el error en sus resultados.
Otra observación es que cuando el valor del error no cambia (linea recta en el grafico) es que sus parametros ya están optimizados, es decir, no puede aprender más.

![test1](/image/Figure_2.png)
![test2](/image/Figure_3.png)
