import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def vector_norm_l0(matrix):
    mask = matrix > 0
    return np.sum(mask, axis=1)


def vector_norm_l1(matrix):
    abs_m = np.abs(matrix)
    return np.sum(abs_m, axis=1)


def vector_norm_l2(matrix):
    return np.sqrt(np.sum(matrix ** 2, axis=1))


def vector_norm_inf(m):
    return np.max(m, axis=1)

def feature_norm_l0(matrix):
    mask = matrix > 0
    return np.sum(mask, axis=0)


def feature_norm_l1(matrix):
    abs_m = np.abs(matrix)
    return np.sum(abs_m, axis=0)


def feature_norm_l2(matrix):
    return np.sqrt(np.sum(matrix ** 2, axis=0))


def feature_norm_inf(m):
    return np.max(m, axis=0)


def feature_mean(dataset):
    return np.nanmean(dataset, axis=0)


def feature_std(dataset):
    return np.nanstd(dataset, axis=0)


def generic_dataset(dim, samples, n_clusters, k, plot=False):
    # Generar clusters con centros random
    clusters = np.random.rand(n_clusters, dim)

    # Alejar los clusters y expandirlos en la cantidad de samples deseadas
    clusters_r = k * np.repeat(clusters, samples//n_clusters, axis=0)

    normal_noise = np.random.normal(loc=0, scale=1, size=(samples, dim))
    dt = clusters_r + normal_noise

    if plot is True:
        if dim == 2:
            plt.scatter(dt[:,0], dt[:,1])
            plt.show()
        elif dim == 3:
            #fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")
            ax.scatter3D(dt[:,0], dt[:,1], dt[:,2])
            plt.show()

    return dt


def add_random_nan(dataset, nan_size):
      
    shape = dataset.shape
    size = dataset.size

    # Generar una lista random de números de tamano "nan_size" comprendidos
    # entre 0 y el size del dataset
    mask = np.random.choice(np.arange(size), int(nan_size), replace=False)
    
    # Convertir esta lista de indices planos al formato del shape del dataset
    mask_unravel = np.unravel_index(mask, shape)
    
    # Reemplazar aquellos indices seleccionados por NaN
    dataset[mask_unravel] = np.nan


def replace_nan_feature_mean(dataset):
    # Calculo la media de cada columna sin considerar los Nan    
    #feature_mean = np.nanmean(dataset, axis=0)
    mean = feature_mean(dataset)
    # Reemplazo los Nan con la media
    dataset = np.where(np.isnan(dataset), mean, dataset)
    return dataset


def pca(dataset, new_dim):
    # Evaluar mean por columna
    mean_columnas = feature_mean(dataset)

    # Calcular la diferencia entre los valores de las columnas y sus meadias

    # Remover la media de las columnas
    dataset_sin_mean = dataset - mean_columnas

    # Calcular covarianza
    dataset_cov = np.cov(dataset_sin_mean.T)

    # autovalor "w" y autovector "v"
    w,v = np.linalg.eig(dataset_cov)

    # Debo ordernar los autovalores de mayor a menor
    arg_sort = np.argsort(w * -1)
    v_sort_max_min = v[:, arg_sort] # Ordeno los autovectores que son columnas por los autovalores

    # Mismo forma de hacer lo mismo
    arg_sort = w.argsort()[::-1] # Estoy inviertiendo todo los indices
    v_sort_max_min = v[:, arg_sort] # Ordeno los autovectores que son columnas

    return np.matmul(dataset_sin_mean, v_sort_max_min[:, :new_dim])


def my_exponential(x, lambda_param=1.0):
    return (- np.log(1-x) / lambda_param)


def k_means(X, n_clusters, iterations=100):
    centroids = np.random.rand(n_clusters, X.shape[1])
    for i in range(iterations):
        centroids, cluster_ids = k_means_loop(X, centroids)
    
    return centroids, cluster_ids


def k_means_loop(X, centroids):
    # find labels for rows in X based in centroids values
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    # recompute centroids
    i = 5
    for i in range(centroids.shape[0]):
        val = X[arg_min == i, :]
        if val.size > 0:
            centroids[i] = np.mean(X[arg_min == i, :], axis=0)

    return centroids, arg_min


if __name__ == '__main__':  
    # Definir los parámetros de entrada
    dim = 4
    sample = 100000
    n_clusters = 4
    nan_size_percent = 0.1

    # Generar un dataset
    dt = generic_dataset(dim=dim, samples=sample, n_clusters=n_clusters, k=100, plot=False)

    # Eliminar el 0.1% de los datos 
    # add_random_nan(dt, dt.size * (nan_size_percent/100.0))

    # # Save array
    # np.save('clase_2.npy', dt, allow_pickle=True)

    # # Load array
    # dt_pickle = np.load('clase_2.npy', allow_pickle=True)

    # # Reemplazar NaN por la media
    # dt = replace_nan_feature_mean(dt)

    # Calcular valores de los features
    print('Feature norm l1', feature_norm_l1(dt))
    print('Feature mean', feature_mean(dt))
    print('Feature std', feature_std(dt))

    # Calcular la VA exponencial
    # exponential_vectorize = np.vectorize(my_exponential)
    # dt_exp = exponential_vectorize(dt)
    # plt.hist(dt_exp)
    # plt.show()

    # Reducir a 2 dimensiones y plotear
    dt2 = pca(dt, 2)

    # Calcular los centros con k-means
    kmeans_centroids, kmeans_cluster_ids = k_means(dt2, 4, 10)
    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dt2)

    # Comparar y plotear
    plt.scatter(dt2[:,0], dt2[:,1])
    plt.scatter(kmeans_centroids[:,0], kmeans_centroids[:,1])
    #plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
    plt.show()

    '''
    Conclusion:
    A pesar de haber agregado ruido en el datasets (NaN reemplazados por mean)
    se puede apreciar a simple vista 4 conjuntos dominantes y el algoritmo
    de k-means los identifica sin problemas
    '''

    # Repetir procedimiento con un dataset cuyos centroidos estén más cerca
    # dt = generic_dataset(dim=dim, samples=sample, n_clusters=n_clusters, k=10, plot=False)
    # dt2 = pca(dt, 2)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dt2)
    # plt.scatter(dt2[:,0], dt2[:,1])
    # plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
    # plt.show()

    '''
    Conclusion:
    Se puede apreciar que al acercar los centroides tenes una gran posibilidad
    de que al menos dos nubes de puntos se junten por lo que 4 clusters
    para separarlos es inecesario cuando los datos están tan solapados
    '''
