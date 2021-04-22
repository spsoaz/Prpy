import numpy as np
import matplotlib.pyplot as plt
import SPSsim as sim
from importlib import reload
from scipy.optimize import curve_fit
import scipy.stats as stats    
reload(sim)
import Phd

# FUnción exponencial
def exponencial(x, lamda):
    return lamda*np.exp(-lamda*x.astype(np.float64))

spikes, cv=sim.poisson_refr(20, ns=1000)
plt.hist(np.diff(spikes), density=True)
y,  edges=np.histogram(np.diff(spikes) , density=True)
x=(edges[1::]+edges[0:-1])/2
params, cov=curve_fit(exponencial, xdata=x, ydata=y)
print("El coeficiente de variación es: ", cv)
print("Los parámetros resultantes del ajuste son:", params)
print("La tasa estimada es: ", np.size(spikes)/(spikes[-1]-spikes[0]))
plt.plot(x, exponencial(x, params[0]), label="fitting")
plt.scatter(x, y, label="data")
plt.legend(loc="best", frameon=False)
plt.show(block=False)


# Ahora prueba con la función  que no tiene retractoricidad
spikes2=sim.poisson_generator(20, 2, 10, 20)
isi=np.diff(spikes2, axis=1).flatten()
y2, edges=np.histogram(isi, density=True)
x2=(edges[1::]+edges[0:-1])/2
params2, cov=curve_fit(exponencial, xdata=x2, ydata=y2)
print("El coeficiente de variación es: ", np.std(isi)/np.mean(isi))
print("Los parámetros resultantes del ajuste son:", params2)
print("La tasa estimada es: ", np.size(spikes2, axis=1)/(spikes2[0, -1]-spikes2[0, 0]))
plt.figure()
plt.hist(isi, density=True)
plt.plot(x2, exponencial(x2, params2[0]), label="fitting")
plt.scatter(x2, y2, label="data")
plt.legend(loc="best", frameon=False)
plt.show(block=False)

Spikes3=[]
for i in range(220):
   Spikes3.append(sim.poisson_generator(20, 2, 10, 1).transpose())

Ntr, CV2_24=Phd.CV2(Spikes3, 0.2,  0.010, 1, 12) 
plt.figure()
plt.plot(np.sum(CV2_24, axis=0)/np.sum(Ntr, axis=0), label="Fit" )
plt.show(block=False)
plt.legend()



### Ahora vienen los tenes con distribución ISI de tipo gamma
print ("Simulaciones con distribuciones de tipo Gamma....")
Ntrials=200
Spikesgamma=[]
ISI=[]
rate = 20
for i in range(Ntrials):
    Spikesgamma.append(sim.gammaSpikes(rate=rate,  ti=2, tend=9))
    ISI.append((np.diff(Spikesgamma[i], axis=0)).flatten())
ISI=np.hstack(ISI)
y2, edges=np.histogram(ISI, density=True)
x2=(edges[1::]+edges[0:-1])/2
plt.figure()
plt.hist(ISI, density=True)
plt.scatter(x2, y2, c="cyan")
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(ISI)
print(fit_alpha, fit_loc, fit_beta)
rv = stats.gamma(fit_alpha, fit_loc, fit_beta)
plt.plot(x2, rv.pdf(x2), 'k-', lw=2, label='fit pdf')
plt.show(block=False)
print("El coeficiente de variación es: ", np.std(ISI)/np.mean(ISI))
mean, var  = stats.gamma.stats(1.99, moments='mv')
print("El coeficiente de variación Teórico es : ",  (var**0.5)/mean)
print("La ventana sugerida para el cálculo es: ", np.mean(ISI)*10 )
Ntr, CV2_24=Phd.CV2(Spikesgamma, 0.2,  0.010, 3, 8) 
plt.figure()
plt.plot(np.nansum(CV2_24, axis=0)/np.nansum(Ntr, axis=0), label="Fit" )
plt.show(block=False)
