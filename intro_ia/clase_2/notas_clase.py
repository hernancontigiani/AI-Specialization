import numpy as np

x = np.array([
            # X1      # X2      # X3
            [1,2,3], [4,5,6], [7,8,9]
            ])
c = np.array([
            [0,0,1], [0,1,1]
            ])

# X --> 3x3
# C --> 2x3
# C_exp --> 2x1x3
# c_x = 2x3x3

expanded_c = c[:, None]

distancia = np.sqrt(np.sum((expanded_c-x)**2, axis=2))
print(distancia)

#  x1           x2         x3
#[[ 3.          8.1240384  13.3041347 ] # distancia del vector X0 a C
#[ 2.44948974  7.54983444 12.72792206]] # distancia del vector X1 a C


vectores_cercanos = np.argmin(distancia, axis=0)  # Buscar los indices menores del vector "distancia" por columna
# esta búsqueda "minimiza" la distancia de X a C.

# axis es la dimension que quiero "comprimir, 
# quiero comprimir las filas --> axis=0 (busco por columna)
# quiero comprimir las columnas --> axis=1 (busco por fila)
# quiero comprimir la dimension --> axis=2

print(vectores_cercanos)


# Generacion de datasets
print("Generacion de datasets")
# A partir de un centroide generar datos
c = np.array([
            [1,0,0,0],
            [0,1,0,0]
            ])
print(c)

k = 100
# Alejar los centroides al multiplicarlos por una constante
c_k = k * c

print(c_k)

n = 4
#np.repeat  --> A partir de una matriz generar una más grande
c_repeat = np.repeat(c_k, n/2, axis=0)
# Como funciona repeate, si x = [x1, x2, x3]
# Si n=2
# repeate = [x1 x1 x2 x2 x3 x3]
print(c_repeat)

# Sumar un valor random normal de media=0 y desv=0
# Sumar ruido blanco
normal_noise = np.random.normal(loc=0, scale=1, size=(n, 4))
c_alt = c_repeat + normal_noise
print(c_alt)

# Iteradores
# np.fromiter(embeddings_gen, structure)
# Son como listas para que no cargan todo en memoria de una


print("Generacion de variables aleatorios en base a un vector uniforme 'U'")
lamb = 1
size = 100

u = np.random.uniform(low=0.0, high=1.0, size=size)
x = - np.log(1-u) / lamb

# si tuviera una funcion de densidad de probabilidad (dpf) --> fx(x) = 3x²
# Fx(x) es la integral de fx(x) = x³
# Fx(x) inversa => X = U^(1/3)
x = np.power(u, 1 /3.0)

print("PCA a mano!!")
# dataset = np.array([
#             [1,10,5,6],
#             [0,1,0,0]
#             ])

# dado el dataset de nxm, donde n = 4 y m = 3 deseo aplicar PCA y reducir a 2 columnas (d=2)

m = 3
n = 4
d = 2
dataset = np.array([
            [0.4,4800,5.5],
            [0.7,12104,5.2],
            [1,12500,5.5],
            [1.5,7002,4.0]
            ])

# Evaluar mean y std de mi dataset por columna
mean_columnas = np.mean(dataset, axis=0)
desv_columnas = np.std(dataset, axis=0)
print("Media de las columnas")
print(mean_columnas)
#print(desv_columnas)

# Calcular la diferencia entre los valores de las columnas y sus meadias

print("Remover la media de las columnas")
dataset_sin_mean = dataset - mean_columnas
print(dataset_sin_mean)

#dataset_cov = np.matmul(np.transpose(dataset_sin_mean),dataset_sin_mean) / n-1
#dataset_cov = np.cov(np.transpose(dataset_sin_mean))
dataset_cov = np.cov(dataset_sin_mean.T)

# autovalor "w" y autovector "v"
w,v = np.linalg.eig(dataset_cov)

# Debo ordernar los autovalores de mayor a menor
arg_sort = np.argsort(w * -1)
v_sort_max_min = v[:, arg_sort] # Ordeno los autovectores que son columnas por los autovalores

# Mismo forma de hacer lo mismo
arg_sort = w.argsort()[::-1] # Estoy inviertiendo todo los indices
v_sort_max_min = v[:, arg_sort] # Ordeno los autovectores que son columnas

resultado = np.matmul(dataset_sin_mean, v_sort_max_min[:, :d])

print("PCA a mano")
print(resultado)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("PCA scikitlearn")
pca = PCA(n_components=2)
x_std = StandardScaler(with_std=False).fit_transform(dataset)
pca.fit_transform(x_std)
print(x_std)