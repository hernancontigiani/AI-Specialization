import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_data():
    path = os.path.dirname(os.path.realpath(__file__))
    posicion_file_path = os.path.join(path, 'posicion.csv')
    velocidad_file_path = os.path.join(path, 'velocidad.csv')
    aceleracion_file_path = os.path.join(path, 'aceleracion.csv')

    posicion_data = np.genfromtxt(posicion_file_path, delimiter=',')
    velocidad_data = np.genfromtxt(velocidad_file_path, delimiter=',')
    aceleracion_data = np.genfromtxt(aceleracion_file_path, delimiter=',')

    time = posicion_data[:,0]

    posicion_data = posicion_data[:,1:4]
    velocidad_data = velocidad_data[:,1:4]
    aceleracion_data = aceleracion_data[:,1:4]

    return time, posicion_data, velocidad_data, aceleracion_data

def get_meassure(posicion_data, velocidad_data, aceleracion_data, type=1, plot=False):

    size = posicion_data.shape[0]
    pos_noise_mean = 0
    pos_noise_std = 10
    speed_noise_mena = 0
    speed_noise_std = 0.2

    if type == 1:

        pos_noise_x = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)
        pos_noise_y = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)
        pos_noise_z = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)

        speed_noise_x = 0
        speed_noise_y = 0
        speed_noise_z = 0

        accel_noise_x = 0
        accel_noise_y = 0
        accel_noise_z = 0

        R = np.zeros(shape=(3,3))
        np.fill_diagonal(R, [pos_noise_std**2, pos_noise_std**2, pos_noise_std**2])

        C = np.zeros(shape=(3,9))
        np.fill_diagonal(C, [1, 1, 1])

    if type == 2:

        uniform_max = np.sqrt((pos_noise_std**2)*12) / 2.0    # mean = 0
        uniform_min = -uniform_max                  # mean  = 0

        pos_noise_x = np.random.uniform(uniform_min, uniform_max, size)
        pos_noise_y = np.random.uniform(uniform_min, uniform_max, size)
        pos_noise_z = np.random.uniform(uniform_min, uniform_max, size)

        speed_noise_x = 0
        speed_noise_y = 0
        speed_noise_z = 0

        accel_noise_x = 0
        accel_noise_y = 0
        accel_noise_z = 0

        R = np.zeros(shape=(3,3))
        np.fill_diagonal(R, [pos_noise_std**2, pos_noise_std**2, pos_noise_std**2])

        C = np.zeros(shape=(3,9))
        np.fill_diagonal(C, [1, 1, 1])

    if type == 3:

        pos_noise_x = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)
        pos_noise_y = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)
        pos_noise_z = np.random.normal(loc=pos_noise_mean, scale=pos_noise_std, size=size)

        speed_noise_x = np.random.normal(loc=speed_noise_mena, scale=speed_noise_std, size=size)
        speed_noise_y = np.random.normal(loc=speed_noise_mena, scale=speed_noise_std, size=size)
        speed_noise_z = np.random.normal(loc=speed_noise_mena, scale=speed_noise_std, size=size)

        accel_noise_x = 0
        accel_noise_y = 0
        accel_noise_z = 0

        R = np.zeros(shape=(6,6))
        np.fill_diagonal(R, [pos_noise_std**2, pos_noise_std**2, pos_noise_std**2,
                             speed_noise_std**2, speed_noise_std**2, speed_noise_std**2])

        C = np.zeros(shape=(6,9))
        np.fill_diagonal(C, [1, 1, 1, 1, 1, 1])


    pos_noise = np.array([pos_noise_x, pos_noise_y, pos_noise_z]).T
    speed_noise = np.array([speed_noise_x, speed_noise_y, speed_noise_z]).T
    accel_noise = np.array([accel_noise_x, accel_noise_y, accel_noise_z]).T

    pos_measure = posicion_data + pos_noise
    speed_measure = velocidad_data + speed_noise
    accel_measure = aceleracion_data + accel_noise

    measure = np.vstack([pos_measure[:,0], 
                        pos_measure[:,1],
                        pos_measure[:,2],
                        speed_measure[:,0],
                        speed_measure[:,1],
                        speed_measure[:,2],
                        accel_measure[:,0],
                        accel_measure[:,1],
                        accel_measure[:,2]
                        ]).T

    if plot is True:
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(posicion_data[:,0], posicion_data[:,1], posicion_data[:,2], c='b')
        ax.scatter(pos_measure[:,0], pos_measure[:,1], pos_measure[:,2], c='r')
        plt.show()

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(velocidad_data[:,0], velocidad_data[:,1], velocidad_data[:,2], c='b')
        ax.scatter(speed_measure[:,0], speed_measure[:,1], speed_measure[:,2], c='r')
        plt.show()

    return measure, R, C


def kalman(time, measure, R, C, real_data):

    # Kalcman Init
    X = np.array([10.7533, 36.6777, -45.1769, 1.1009, -17.0, 35.7418, -5.7247, 3.4268, 5.2774]).reshape(-1, 1) # 9x1
    P = np.zeros(shape=(9,9))
    np.fill_diagonal(P, [100, 100, 100, 1, 1, 1, 0.01, 0.01, 0.01])
    Q = 0.3 * np.identity(9)
    last_time = 0
    error_m = np.zeros(shape=(len(time), 9))
    predicciones = np.zeros(shape=(len(time), 9))
    predicciones[0, :] = X.T
    

    # Ciclo Kalman
    for i in range(1, len(time)):
        t = time[i]
        dt = t - last_time

        A_dt = np.zeros(shape=(9,9))
        ones = np.ones(6)
        np.fill_diagonal(A_dt[:, 3:], ones)

        A_dt2 = np.zeros(shape=(9,9))
        ones = np.ones(3)
        np.fill_diagonal(A_dt[:, 6:], ones)

        A = np.identity(9) + A_dt * dt + A_dt2 * (dt**2)

        # Predicción
        X_propagated = np.matmul(A, X)
        P_propagated = np.matmul(np.matmul(A, P), A.T) + Q

        # Correccion
        inv = np.linalg.inv(np.matmul(np.matmul(C, P_propagated), C.T) + R)
        K = np.matmul(np.matmul(P_propagated, C.T), inv)

        # Medicion
        medicion = measure[i, :]
        y = np.matmul(C, medicion).reshape(-1, 1)

        z = y - np.matmul(C, X_propagated).reshape(-1, 1)

        # Estimacion
        X_new = X_propagated + np.matmul(K, z)

        # Verifico el error entre lo estimado y lo real
        error = (real_data[i, :] - X_new.T)
        error_m[i, :] = error

        # Termina la iteracion
        predicciones[i, :] = X_new.T
        X = X_new
        last_time = t

    error_pos = np.sqrt(error_m[:,0]**2 + error_m[:,1]**2 + error_m[:,2]**2)
    return error_pos, predicciones


if __name__ == "__main__":

    time, posicion_data, velocidad_data, aceleracion_data = get_data()

    real_data = np.vstack([posicion_data[:,0],
                posicion_data[:,1],
                posicion_data[:,2],
                velocidad_data[:,0],
                velocidad_data[:,1],
                velocidad_data[:,2],
                aceleracion_data[:,0],
                aceleracion_data[:,1],
                aceleracion_data[:,2]
                ]).T

    measure, R, C = get_meassure(posicion_data, velocidad_data, aceleracion_data, 1, False)
    error_pos_1, predicciones_1 = kalman(time, measure, R, C, real_data)

    error_1 = real_data - predicciones_1

    fig = plt.figure()
    fig.suptitle('Error en la posición punto 1', fontsize=16)
    ax = fig.add_subplot()
    ax.plot(error_1[:, 0], c='b', label='Pos_x')
    ax.plot(error_1[:, 1], c='r', label='Pos_y')
    ax.plot(error_1[:, 2], c='m', label='Pos_z')
    ax.set_ylabel("error[metros]")
    ax.set_xlabel("ciclos kalman[1seg c/u]")
    ax.legend()
    plt.show()

    measure, R, C = get_meassure(posicion_data, velocidad_data, aceleracion_data, 2)
    error_pos_2, predicciones_2 = kalman(time, measure, R, C, real_data)

    error_2 = real_data - predicciones_2

    fig = plt.figure()
    fig.suptitle('Error en la posición punto 2', fontsize=16)
    ax = fig.add_subplot()
    ax.plot(error_2[:, 0], c='b', label='Pos_x')
    ax.plot(error_2[:, 1], c='r', label='Pos_y')
    ax.plot(error_2[:, 2], c='m', label='Pos_z')
    ax.set_ylabel("error[metros]")
    ax.set_xlabel("ciclos kalman[1seg c/u]")
    ax.legend()
    plt.show()
    
    measure, R, C = get_meassure(posicion_data, velocidad_data, aceleracion_data, 3)
    error_pos_3, predicciones_3 = kalman(time, measure, R, C, real_data)

    error_3 = real_data - predicciones_3

    fig = plt.figure()
    fig.suptitle('Error en la posición punto 3', fontsize=16)
    ax = fig.add_subplot()
    ax.plot(error_3[:, 0], c='b', label='Pos_x')
    ax.plot(error_3[:, 1], c='r', label='Pos_y')
    ax.plot(error_3[:, 2], c='m', label='Pos_z')
    ax.set_ylabel("error[metros]")
    ax.set_xlabel("ciclos kalman[1seg c/u]")
    ax.legend()
    plt.show()

    fig = plt.figure()
    fig.suptitle('Error en la posición', fontsize=16)
    ax = fig.add_subplot()

    ax.plot(error_pos_1, c='b', label='1) Pos Noise Gauss')
    ax.plot(error_pos_2, c='r', label='2) Pos Noise Uniform')
    ax.plot(error_pos_3, c='m', label='3) Pos+Speed Noise Gauss')
    ax.set_ylabel("error[metros]")
    ax.set_xlabel("ciclos kalman[1seg c/u]")
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot(real_data[:,0], real_data[:,1], real_data[:,2], c='g', label='Datos reales')
    ax.plot(predicciones_1[:,0], predicciones_1[:,1], predicciones_1[:,2], c='b', label='1) Pos Noise Gauss')
    ax.plot(predicciones_2[:,0], predicciones_2[:,1], predicciones_2[:,2], c='r', label='2) Pos Noise Uniform')
    ax.plot(predicciones_3[:,0], predicciones_3[:,1], predicciones_3[:,2], c='m', label='3) Pos+Speed Noise Gauss')
    ax.legend()
    plt.show()


