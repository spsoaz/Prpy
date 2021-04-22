#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------      Funciones útles          ------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from numpy import loadtxt, concatenate, shape, unique, mean, std, arange
from numpy import sum as npsum
from numba import njit, vectorize, float64, int32, guvectorize
import numexpr as ne
from scipy.optimize import curve_fit
#from  Funciones_AlinVsSigm import linearfit
import ray
ray.init(ignore_reinit_error=True)


lengthRef=95  #Indicador del ancho de ventana para estandarización 
#Diccionario indicador de las cabeceras
headers={"Serie":0, "Profundida":1, "Elec":2, "Uni":3, "setN":4, "SetId":5, "CR":6, "Elec_cuidado":7, "Adaptador":8, "3b":9, "Hits":10, "Amp":11, "Trial":12, "Data":13}

def estandariza(data):
    series33=unique(data[:, headers["Serie"]])
    for i in series33:
        for elec in range(1, 8):
            mask=(data[:, headers["Serie"]]==i)*(data[:, headers["Elec"]]==elec)
            unidades=unique(data[mask, headers["Uni"]])
            for un in unidades:
                msk2=data[:, headers["Uni"]]==un
                mask2=mask*msk2
                media=mean((data[mask2, headers["Data"]:headers["Data"]+lengthRef]).flatten())
                sigma=std((data[mask2, headers["Data"]:headers["Data"]+lengthRef]).flatten())
                data[mask2, headers["Data"]::]=(data[mask2, headers["Data"]::]-media)/sigma


def fitmatlab(x):
    from numpy import log
    return ne.evaluate("9.904*log(96.58*x+12.75) + 4.749")

#@njit()
def linear(x, m, b):
    return m*x + b

#@njit("float64(float64, float64, int32 ) ")
def  linearfit( xdata, ydata,   N):
    """
    Esta función devuelve los parámetros de un ajuste lineal mediante el uso de mínimos cuadrados, la función         
    requiere tres entradas:
    *  xdata: Abscsa de los datos
    *  ydata: Ordenada de los datos
    *  params: arreglo donde se colocarán la pendiente y la ordenada al origen.    
    * N: El número de elementos en cada uno de los arreglos de datos.
    """
    meanx=0 
    meany=0
    num=0
    den=0
    i=0;
    params=np.zeros((2), dtype=np.float64)
    for i in range(N):
        meanx+=xdata[i]
        meany+=ydata[i]
    meany/=N
    meanx/=N
    for i in range(N):
        num+=(ydata[i]*(xdata[i] - meanx)  )
        den+=(xdata[i]*(xdata[i] - meanx)  )          
    params[0]=num/den
    params[1]=meany-meanx*params[0]
    return params
    



#Esta función no es conveniente paralelizarla debido a que  entrará en el ciclo a paralelizar con Ray
@vectorize(["float64(float64, float64, float64, float64, float64)"], )
def sigmoidal(x, a, b, c ,d):
    return (a-b)/(1 + np.exp((d-x)/c) ) + b

#@njit()        
def Transform_Data(headers_Data, RR, wnd_base, wnd_fin):
    '''
      Esta función pasa de la tasa de ventanas de 50 ms a pasos de 10 a ventanas de 200ms a pasos de 10 ms.
      
      Perdata: Matriz donde se almacenaran los nuevos valores, nota que esta matriz es modificada permanentemente
                     dentro de la función.
      bd: Indica dónde inicia la columna de tasas de disparo en  RR
      RR: Matriz de datos originales donde aparece la tasa con ventanas de 50 ms avanzando  cada 10 ms.
              Esta matriz tiene el orden ndicado en el diccionario headers.
    '''
    #Cuánto es necesario 
    c=int(wnd_fin//wnd_base)
    rdat, cdat=np.shape(RR)
    #print(rdat, cdat)
    PerData=np.zeros(( rdat,  cdat-(c+1)*(c-1)-1), dtype=np.float32)
    #r, c=PerData.shape
    #for renglon in range(rdat):
    col_combine=np.array([(c+1)*i  for i in range(c) ]) + headers_Data
    for column in range(headers_Data, cdat-(c+1)*(c-1)-1):
            #print(col_combine)
            PerData[:,  column]=npsum(RR[:, col_combine], axis=1)
            col_combine+=1
    PerData[:, 0:headers_Data]=RR[:, 0:headers_Data]
    return PerData
   
   
def   permuta_Todo(nperm, m, beta, tasa, xampTac, xampAud, RMSElineal, RMSEsigmoidal):
    """
    """
    for i in range(nperm):  #Ejecución de ajustes permutados, este es el que se va a paralelizar.
                model, cov=curve_fit(linear, xdata=xampTac, ydata=tasa)        
                modelsig, covsig=curve_fit(sigmoidal, xdata=xampTac, ydata=tasa, maxfev = 8000, p0=(5, 2, 1, 10), bounds=((0.1, -5, 0.1, 7), (50, 5, 30, 15)))
                lintmpRMSE=np.sum((tasa-linear(xampTac, model[0], model[1]))**2 )
                sigmtmpRMSE=np.sum((tasa-sigmoidal(xampTac, modelsig[0], modelsig[1], modelsig[2],  modelsig[3]))**2 )
                if lintmpRMSE<=RMSElineal:
                    plin+=1   #Cuenta la p de que la otra sea igual o mayor en su pendiente.
                if sigmtmpRMSE<=RMSEsigmoidal:
                    psig+=1
                if np.abs(m)<=np.abs(model[0]):
                    mp+=1
                if np.abs(beta)>=np.abs(modelsig[2]):
                    betap+=1
            # Normalización con respecto al número de permutaciones
    plin/=nperm
    psig/=nperm  
    mp/=nperm
    betap/=nperm  
    return plin, psig, mp, betap


def   permuta(tasa, xampTac):
    """
    """
    model_lin=np.zeros((2), dtype=np.float64)
    np.random.shuffle(xampTac)
    #model_lin, cov=curve_fit(linear, xdata=xampTac, ydata=tasa)        
    model_lin=linearfit(xampTac, tasa,  np.size(tasa))        #LLama a función optimizada.  
    lintmpRMSE=np.sum((tasa-linear(xampTac, model_lin[0], model_lin[1]))**2 )
    modelsig, covsig=curve_fit(sigmoidal, xdata=xampTac, ydata=tasa, maxfev = 8000, p0=(5, 2, 1, 10), bounds=((0.1, -5, 0.1, 7),(50, 5, 30, 15)))
    lintmpRMSE=np.sum((tasa-linear(xampTac, model_lin[0], model_lin[1]))**2 )
    sigmtmpRMSE=np.sum((tasa-sigmoidal(xampTac, modelsig[0], modelsig[1], modelsig[2],  modelsig[3]))**2 )
    return model_lin[0], lintmpRMSE, modelsig[2], sigmtmpRMSE

@ray.remote
def permuta_ray(tasa, xampTac):
    return permuta(tasa, xampTac)

def permutaciones(nperm, m, beta, tasa, xampTac,  RMSElineal, RMSEsigmoidal):
        refs =[permuta_ray.remote(tasa, xampTac) for _ in range(1, nperm+1)]
        resultados=ray.get(refs)
        resultados=np.vstack(resultados)
        pm=(np.sum(np.abs(resultados[:, 0])<=m))/nperm
        pbeta=(np.sum(np.abs(resultados[:, 2])<=beta))/nperm
        pRMSElineal=(np.sum(resultados[:, 1]<=RMSElineal))/nperm
        pRMSEsigm=(np.sum(resultados[:, 3]<=RMSEsigmoidal))/nperm
        return pm, pbeta, pRMSElineal, pRMSEsigm

def permutaciones_noray(nperm, m, beta, tasa, xampTac,  RMSElineal, RMSEsigmoidal):
        resultados =[permuta(tasa, xampTac) for _ in range(1, nperm+1)]
        resultados=np.vstack(resultados)
        pm=(np.sum(np.abs(resultados[:, 0])<=m))/nperm
        pbeta=(np.sum(np.abs(resultados[:, 2])<=beta))/nperm
        pRMSElineal=(np.sum(resultados[:, 1]<=RMSElineal))/nperm
        pRMSEsigm=(np.sum(resultados[:, 3]<=RMSEsigmoidal))/nperm
        return pm, pbeta, pRMSElineal, pRMSEsigm

