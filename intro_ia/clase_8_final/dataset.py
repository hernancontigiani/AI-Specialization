import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

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


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        return data
        # np.fromiter

    def shape(self):
        return self.dataset.shape

    def add(self, offset):
        self.dataset += offset

    def split(self, percentage):
        # retornar train y test segun el %
        dim = self.dataset.shape[1]
        if dim > 2:
            X = self.dataset[:,0:(dim-1)]
        else:
            X = self.dataset[:,0]
        y = self.dataset[:,dim-1]
        #train_test_split(X, y, test_size=(1-percentage))
        #X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(1-percentage))

        # Gennero lista de indices desordenados de 0 a n
        permuted_idxs = np.random.permutation(X.shape[0])

        # Genero la lista de indices para el dataset train
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        # Los que quedan son la lista de indices para el dataset test
        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def split_static(X, y, percentage):

        # Gennero lista de indices desordenados de 0 a n
        permuted_idxs = np.random.permutation(X.shape[0])

        # Genero la lista de indices para el dataset train
        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        # Los que quedan son la lista de indices para el dataset test
        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


class CsvData(Data):
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        data = data[1:,1:]
        return data
        # np.fromiter