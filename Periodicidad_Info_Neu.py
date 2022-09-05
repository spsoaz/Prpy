# Bloque de código
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from numba import njit
import Extensiones as Pext
from scipy.signal.windows import tukey
from numba import njit
from os import name as nameos

def loaddata(serie_i, elec, unit, order, tcut=(-1.5, 2.0)):
    if nameos=="posix":
        if serie_i>100:   ## Mono 32
            path="/home/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s/RR032%d_002/RR032%d_002"%(serie_i, serie_i)
        else:             ## Mono 32
            path="/home/sparra/AENHA/Database_RR033/Text_s/RR0330%d_00%d/RR0330%d_00%d"%(serie_i, order, serie_i, order)
    else:            
        if serie_i>100:   ## Mono 32
            path="D:\\BaseDatosKarlitosNatsushiRR032\\Text_s\\RR032%d_002\\RR032%d_002"%(serie_i, serie_i)
        else:             ## Mono 32
            path="D:\\Database_RR033\\Text_s\\RR0330%d_00%d\\RR0330%d_00%d"%(serie_i, order, serie_i, order)

    try:
        A=np.loadtxt(path+ "_e%d_u%d.csv"%(elec, unit), delimiter=",", usecols=(0))
        delimiter=","
    except IOError:
        print("El archivo %s no existe"%(path + "_e%d_u%d.csv"%(elec, unit)))
        return -1
    except:
        A=np.loadtxt(path+ "_e%d_u%d.csv"%(elec, unit), delimiter="\t", usecols=(0))
        delimiter="\t"
    spikes=Pext.SLoad(path+ "_e%d_u%d.csv"%(elec, unit), delimiter)
    try:
        psyc=np.loadtxt(path + "_Psyc.csv", delimiter=",")
    except:
        psyc=np.loadtxt(path + "_Psyc.csv", delimiter="\t")
    psyc=psyc[(A-1).astype(np.int32), :]
    try:
        times=np.loadtxt(path + "_T.csv", delimiter=",")
    except:
        times=np.loadtxt(path + "_T.csv", delimiter="\t")
    times=times[(A-1).astype(np.int32), :]
    spikesshort=[]
    for trial_i in range(len(A)):
        mask=(spikes[trial_i][1::]>=(times[trial_i, 6])+tcut[0] )*(spikes[trial_i][1::]<(times[trial_i, 6]+ tcut[1]) )
        spikesshort.append(spikes[trial_i][1::][mask]-times[trial_i, 6])
    return times, psyc, spikesshort

class neuromParams:
    """
    Clase de solo atributos con los siguientes:
      ind: booleano que indica si se actualizarán los elementos dentro de optimumC
      bw: Se refiere al ancho de cada bin
      BE: Se refiere a las esquinas que delimitan cada bin o ventana.
    """
    def __init__(self, ini, binwidth, BinEdges, nbins=12):
        self.ind=ini
        self.bw=binwidth
        self.BE=BinEdges
        self.nbins=nbins
    def update(self, ini, binwidth, BinEdges, nbins):
        self.ind=ini
        self.bw=binwidth
        self.BE=BinEdges
        self.nbins=nbins

def optimumC(tasa, Psicof,  calcbins, return_histogram=False):
    """
    Created on Mon Feb 25 13:35:21 2019
    
    @author: sparra
    
    Esta función realiza la neurometría basada en el criterio óptimo. Como parámetros de entrada requiere:
    * tasa: Un nupy array con las distintas tasas
    * amp:  Un numpy array con la amplitud correspondiente al vector de tasas
    * return_histogram, Es un escalar booleano que permite indicar si se devolverá el histograma de los valores o no. Por defecto NO
    * calcbins, es una clase de solo atributos, estos últimos son los siguientes:
        ind: escalar que indica si se calcularan los bines. SI es cierto se actualizan los valores del elemento de clase calcbins
        si ind es falso utiliza los valores pasados como atributos.
        
    Esta función hace uso de la función de psicofísica, la cual devuelve los resultados de psicofísica del día, para ellos
    sólo requiere del nombre del archivo.

    Esta función devuelve una lista con la neurometría, el primer elemento tiene una matriz con la iformación de la neurometría 
    táctil, el segundo elemento contiene la neurometría auditiva y el tercero el resultado del conteo del fr para cada modalidad.


    """
    from numpy import zeros, loadtxt, size, unique, sum, size, argmax, max, min, array, linspace, abs, histogram
    from numpy import float32, argmin

    amplitudes, counts=np.unique(Psicof, return_counts=True)
    if calcbins.ind:
        None
    else:  
        maximum=np.max(tasa)
        minimum=np.min(tasa)
        X=linspace(minimum, maximum, calcbins.nbins)
        binwidth=X[1]- X[0]
        calcbins.update(False, binwidth, X, calcbins.nbins)
    #Starting with histogram of zeros        
    Histograms=zeros( (calcbins.nbins, len(amplitudes)), dtype=float32)
    index=0
    for amp_i in amplitudes:
        indices=np.where(Psicof==amp_i)[0]
        Histograms[:, index], _=histogram(tasa[indices], bins=calcbins.BE)
        Histograms[:, index]/=len(indices)
        index+=1
   # Criterio óptimo para la neurometría táctil
    Hits=zeros((calcbins.nbins, 1), dtype=float)
    for co in range(1, calcbins.nbins+1):
        for amp in range(0, len(amplitudes)):
            if amp==0:
                Hits[co-1]+=(sum(Histograms[0:co, amp] ) )*counts[amp]/sum(counts)
            else:
                Hits[co-1]+=(sum(Histograms[co::, amp] ) )*counts[amp]/sum(counts)    
    # Here begin the calculation for Optimum criterion 
    c=argmax(Hits.astype(np.float32))
    neur=zeros((1, len(amplitudes)), dtype=float)
    for amp in range(0, len(amplitudes)):
        if amp==0:
            neur[0, amp]+=(sum(Histograms[:c, amp] ) )
        else:
            neur[0, amp]+=(sum(Histograms[c::, amp] ) )    
    neur[0, 0]=1-neur[0, 0]  #Probabilidad de falla para la amplitud cero.
    # Until here, was calculated neurometric by using only two possible responses 
    #print(counts/sum(counts))
    if return_histogram==0:        
        return  neur, Hits, c, 
    else:
        return neur,  Hits, c, Histograms
    
def sigmoidal(X,a,b,c,d):
    from numpy import exp
    return (a-b)/(1+exp((d-X)/c))+b
    
def calculate_probstim(Data):
    amp, Ps=np.unique(Data, return_counts=True)
    return amp, Ps/len(Data)

@njit()
def histograms(Data, BinEdges, Amplitudes, amp):
    PrIamp=np.zeros((len(BinEdges)-1 ,len(amp)), dtype=np.float32)
    for indice, amp_i in enumerate(amp):
        pos=np.where(Amplitudes==amp_i)[0]
        PrIamp[:, indice], _=np.histogram(Data[pos], bins=BinEdges)
        PrIamp[:, indice]=PrIamp[:, indice]/len(pos)
    return PrIamp

@njit()
def Minf(PrIs, Pr, Ps, nr, nc):
    informacion=0
    for s in range(nc):
        for r in range(nr):
            if Pr[r]>0 and PrIs[r, s]>0:
                informacion+=PrIs[r,s]*Ps[s]*np.log2(PrIs[r,s]/Pr[r])
    return informacion

@njit()
def Inf_time(datamatrix, lent,  BinEdges, amplitudes, amp, PsT, nbins):
    I_t=np.zeros(lent, dtype=np.float32)
    for t_i in range(lent): 
        PrIampT=histograms(datamatrix[:, t_i], BinEdges, amplitudes, amp)
        Pr, _=np.histogram(datamatrix[:, t_i], bins=BinEdges)
        Pr=Pr/len(amplitudes)
        I_t[t_i]=Minf(PrIampT, Pr, PsT, nbins, len(amp))
    return I_t

## Bias estimado mediante la aproximación asintótica (Se requieren muchos ensayos de modo que cada posible respuesta aparezca)
@njit()
def Bias_PT(data, BinEdges, amplitudes, amp):
    nr=len(data)
    nc=len(data[0, :])
    bias=np.zeros((nc), dtype=np.float64)
    for i in range(nc):
        Pr, _=np.histogram(data[:, i], BinEdges)
        Histogramas=histograms(data[:, i], BinEdges, amplitudes, amp)
        bias[i]=np.sum(np.sum((Histogramas>0), axis=0)-1) - np.sum(Pr>0) +1
    bias=bias/(2*nr*np.log(2))
    return bias