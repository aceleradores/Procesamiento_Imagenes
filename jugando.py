import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def funcion_ajuste(x, A, sigma, offset,centro):
    return A/sigma * np.exp(-1/2*((x-centro)/sigma)**2) + offset


def resumen(names,values,diagsqrt):
    for i in range(len(names)):
        print(names[i],": \t","{0:0.5f}".format(values[i]),"\t +/- ","{0:0.5f}".format(diagsqrt[i]))

parametros=["Amplitud","sigma   ","offset  ","centro  "]

# datos inventados
x_medido=np.linspace(-5,10,82)
y_medido=funcion_ajuste(x_medido+np.random.rand(np.size(x_medido))-.5,3,2,.5,4)


popt, pcov = curve_fit(funcion_ajuste, x_medido, y_medido)

resumen(parametros,popt,np.diag(pcov)**(1/2))

x_ajuste=x_medido
y_ajuste=funcion_ajuste(x_ajuste,*popt)

plt.figure()
plt.scatter(x_medido,y_medido)
plt.plot(x_medido,y_ajuste)
plt.show()