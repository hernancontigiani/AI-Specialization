
import numpy as np
import matplotlib.pyplot as plt

N = 500

z_0_mean = 5
z_0_std = 15
z_0_p = 0.25

z_1_mean = 10
z_1_std = 2
z_1_p = 0.75

z_0 = np.random.normal(loc=z_0_mean, scale=z_0_std, size=int(N*z_0_p))
z_1 = np.random.normal(loc=z_1_mean, scale=z_1_std, size=int(N*z_1_p))

plt.figure()
plt.hist(z_0, 40)
plt.hist(z_1, 40)
plt.show()

# normal multivariable
# numpy.random.multivariate_normal
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
# Es para generar variables random de la normal

# scipy.stats.multivariate_normal
# from scipy.stats import multivariate_normal
# var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
# var.pdf([1,0])