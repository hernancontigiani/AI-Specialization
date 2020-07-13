'''
1. Escribir una clase dataset que levante los datos en un numpy array estructurado
2. La clase tiene que tener un metodo para separar los datos en 80% train y 20% test
3. Hacer una clase que implemente regresion lineal
4. Regresion lineal con b
5. Prediccion
6. Graficos
7. ECM --> MCE
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

'''
structure = [('word', np.dtype('U' + str(cls.WORD_MAX_SIZE))),
             ('embedding', np.float32, (cls.N_FEATURES,))]
structure = np.dtype(structure)
# load numpy array from disk using a generator
with open(cls.WORD_TO_VEC_MODEL_TXT_PATH, encoding="utf8") as words_embeddings_txt:
    embeddings_gen = ((line.split()[0], line.split()[1:])
                      for line in words_embeddings_txt)
    embeddings = np.fromiter(embeddings_gen, structure)
'''

# https://cmdlinetips.com/2020/03/linear-regression-using-matrix-multiplication-in-python-using-numpy/

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)
        
    def _build_dataset(self, path):
        # usar numpy esctructurado
        data = np.genfromtxt(path, delimiter=',')
        data = data[1:,1:]
        return data
        # np.fromiter

    def split(self, percentage):
        # retornar train y test segun el %
        X = self.dataset[:,0]
        y = self.dataset[:,1]
        #train_test_split(X, y, test_size=(1-percentage))
        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=(1-percentage))
        return X_train, X_test, y_train, y_test


class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        # train model
        return NotImplemented

    def predict(self, X):
        # retorna y hat
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        W = Y.mean()
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y hat a partir de X e W
        return np.ones(len(X)) * self.model


class LinearRegresion(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        W = X.T.dot(Y) / X.T.dot(X)
        self.model = W

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y hat a partir de X e W
        return self.model * X


class LinearRegresionWithB(BaseModel):

    def fit(self, X, Y):
        # Calcular los W y guadarlos en el modelo
        X_exp = np.vstack((X, np.ones(len(X)))).T

        W = np.linalg.inv(X_exp.T.dot(X_exp)).dot(X_exp.T).dot(Y)
        # W[0] es la pendiente m
        # W[1] es la ordenada al origen
        self.model = W
        

    def predict(self, X):
        # usar el modelo (self.model) y predecir
        # y hat a partir de X e W
        # 
        X_exp = np.vstack((X, np.ones(len(X)))).T
        y_hat = X_exp.dot(self.model)
        return y_hat


class Metric(object):
    def __call__(self, target, prediction):
        # target --> Y
        # prediction --> Y hat
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        # Implementar el error cuadratico medio MSE
        return np.square(target-prediction).mean()


if __name__ == '__main__':

    dataset = Data('income.data.csv')

    X_train, X_test, y_train, y_test  = dataset.split(0.8)
   
    lr = LinearRegresion()
    lr.fit(X_train, y_train)

    lrb = LinearRegresionWithB()
    lrb.fit(X_train, y_train)

    ct = ConstantModel()
    ct.fit(X_train, y_train)

    lr_y_hat = lr.predict(X_test)
    lrb_y_hat = lrb.predict(X_test)
    ct_y_hat = ct.predict(X_test)

    mse = MSE()
    lr_mse = mse(y_test, lr_y_hat)
    lrb_mse = mse(y_test, lrb_y_hat)
    ct_mse = mse(y_test, ct_y_hat)

    x_plot = np.linspace(0,10,10)
    lr_y_plot = lr.model * x_plot

    x_exp = np.vstack((x_plot, np.ones(len(x_plot)))).T
    lrb_y_plot = np.sum(lrb.model * x_exp, axis=1)

    plt.scatter(X_train, y_train, color='b', label='dataset')
    plt.plot(x_plot, lr_y_plot, color='m', label=f'LinearRegresion(MSE={lr_mse:.3f})')
    plt.plot(x_plot, lrb_y_plot, color='r', label=f'LinearRegresionWithB(MSE={lrb_mse:.3f})')
    plt.plot(X_test, ct_y_hat, color='g', label=f'ConstantModel(MSE={ct_mse:.3f})')
    plt.legend()
    plt.show()

    # verticalstack o horitonzalstack para agregar la columna de unos

    