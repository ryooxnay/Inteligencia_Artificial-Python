import tensorflow as tf #pip install --upgrade --user tensorflow
import numpy as np
import matplotlib.pyplot as plt
 #'' % \

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit =  np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss ='mean_squared_error' 
)
#Emezamos a entrenar el algoritmo
print("Empezamos con el entrenamiento..... ")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose = False)  #epochs--> Para decir cuanas veces se tiene que entrenar el metodo
#verbose  --->  Para que no imprima mucho mugrero
print("Modleo entrenado!!")
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion!!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + "fahrenhit!!")
