import numpy as np
import matplotlib.pyplot as plt

import model
import dataset as dtlib


if __name__ == '__main__':

    # Definir los parámetros de entrada
    dim = 4
    sample = 100000
    n_clusters = 4
    nan_size_percent = 0.1

    # Generar un dataset
    dt = dtlib.generic_dataset(dim=dim, samples=sample, n_clusters=n_clusters, k=100, plot=False)

    #Eliminar el 0.1% de los datos 
    dtlib.add_random_nan(dt, dt.size * (nan_size_percent/100.0))

    # # Save array
    # np.save('clase_2.npy', dt, allow_pickle=True)

    # # Load array
    # dt_pickle = np.load('clase_2.npy', allow_pickle=True)

    # Reemplazar NaN por la media
    dt = dtlib.replace_nan_feature_mean(dt)

    # Calcular valores de los features
    print('Feature norm l1', dtlib.feature_norm_l1(dt))
    print('Feature mean', dtlib.feature_mean(dt))
    print('Feature std', dtlib.feature_std(dt))

    # Calcular la VA exponencial
    # exponential_vectorize = np.vectorize(my_exponential)
    # dt_exp = exponential_vectorize(dt)
    # plt.hist(dt_exp)
    # plt.show()

    # Reducir a 2 dimensiones y plotear
    dt2 = dtlib.pca(dt, 2)

    # Calcular los centros con k-means
    kmeans_centroids, kmeans_cluster_ids = model.k_means(dt2, n_clusters, 100)
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
    # dt = dtlib.generic_dataset(dim=dim, samples=sample, n_clusters=n_clusters, k=10, plot=False)
    # dt2 = dtlib.pca(dt, 2)
    #kmeans_centroids, kmeans_cluster_ids = model.k_means(dt2, n_clusters, 100)
    # plt.scatter(dt2[:,0], dt2[:,1])
    #plt.scatter(kmeans_centroids[:,0], kmeans_centroids[:,1])
    # plt.show()

    '''
    Conclusion:
    Se puede apreciar que al acercar los centroides tenes una gran posibilidad
    de que al menos dos nubes de puntos se junten por lo que 4 clusters
    para separarlos es inecesario cuando los datos están tan solapados
    '''