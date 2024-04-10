import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parámetros de las distribuciones de Poisson
lambda1 = 3
lambda2 = 5

# Simular 1000 muestras de X e Y
num_samples = 1000
X = np.random.poisson(lambda1, num_samples)
Y = np.random.poisson(lambda2, num_samples)

# Calcular la variable Z = X - Y
Z = X - Y

# Graficar la distribución de Z
plt.figure(figsize=(8, 6))
plt.hist(Z, bins=20, density=True, edgecolor='black')

# Calcular la distribución de probabilidad teórica
z_values = np.arange(min(Z), max(Z) + 1)
p_z = poisson.pmf(z_values, lambda1 - lambda2)

# Graficar la distribución teórica
plt.plot(z_values, p_z, 'r-', lw=2, label='Distribución teórica')
plt.xlabel('Valor de Z')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución de la variable Z = X - Y')
plt.legend()
plt.show()