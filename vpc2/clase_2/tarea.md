# Clase 2 - Tarea de VPC2

## Ejer1 - Pool vs No Pool
En pool vs no pool agregar uno o dos bloques más con y sin pool para ver si hay diferencia.\
Se tuvo que agregar todas las capas posibles para lograr una diferencia entre pool y no pool en el entrenamiento. Link al colab:
Colab: [Con_pool_vs_sin_pool](https://colab.research.google.com/drive/1XlRCUGX35hj6Om16aKTwWgIo_gRI8R-n?usp=sharing)

## Ejer2 - Data augmentation
Entrenar la red para perros y gatos y ver qué accuracy pueden obtener, comparar contra data augmentation.\
Con 6 iteraciones el modelo sin augemntation pareciera ser que diera mejor resultado pero en realidad ya comenzaca a relizar overfitting, y la accuracy de validacion rondaba los 0.68. En cambio, al auemntar el set de datos con rotación y zoom el modelo no overfittea, y el accuracy de entrenamiento y validación van de la mano aprox 0.705. Link al colab:
Colab: [Dataset_y_data_augmentation](https://colab.research.google.com/drive/1A-n0Y9BsfGZrdW7arNJWjvWDbQ9t5cA5?usp=sharing)

## Ejer3 - LeNet-5
Implementar LeNet-5 y utilizarla en MNIST
Colab: [LeNet5_minst](https://colab.research.google.com/drive/1xbwqBvFefSsJurTzzbDUfn8_dZ42Pqnt?usp=sharing)