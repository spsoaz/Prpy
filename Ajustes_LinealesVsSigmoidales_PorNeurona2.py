#Carga librerías
from numpy import loadtxt, concatenate, shape, unique, mean, std, arange
from numpy import sum as npsum
import numpy as np
import numba as nb
import numexpr as ne
import matplotlib.pyplot as plt
import pickle
import json
import matplotlib.cm as cm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
cmap=cm.get_cmap("tab20")
import Funciones_AjustesLinealesVsSigmoidales_PorNeourona as ALS
import Funciones_AlinVsSigm as ALSexp
from time import time
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
nperm=10 #1000   # Num de permutaciones
#----------------------------------          Define constantes para el manejo de archivos   ----------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#Diccionario indicador de las cabeceras
headers={"Serie":0, "Profundida":1, "Elec":2, "Uni":3, "setN":4, "SetId":5, "CR":6, "Elec_cuidado":7, "Adaptador":8, "3b":9, "Hits":10, "Amp":11, "Trial":12, "Data":13}

#por neurona (-1+0.05)/0.010   (shift+wnd)/step
#--------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------      Lectura de archivos  ------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
root="/home/sparra/MEGA/Prjl"
#root="/home/sparra/Prjl"
R33=loadtxt(root+"/RR0330Tasa_individual.csv", delimiter=',')
RR032_a=loadtxt(root+"/RR032Tasa_individual_RR032P1.csv", delimiter=',')
RR032_b=loadtxt(root+"/RR032Tasa_individual_RR032P2.csv", delimiter=',')
RR032_c=loadtxt(root+"/RR032Tasa_individual_RR032P3.csv", delimiter=',')
R32=concatenate((RR032_a, RR032_b, RR032_c), axis=0)

neuronas32=np.unique(R32[:, [headers["Serie"], headers["Elec"], headers["Uni"] ]], axis=0)
neuronas33=np.unique(R33[:, [headers["Serie"], headers["Elec"], headers["Uni"] ]], axis=0)

Tasas=np.concatenate((R32, R33), axis=0)
#Adapatación de las tasas a la venatana de interés.
Tasas=ALS.Transform_Data(headers["Data"], Tasas, 50, 200)   
ALS.estandariza(Tasas)

neuronas=np.concatenate((neuronas32, neuronas33), axis=0)
del RR032_a, RR032_b, RR032_c, R32, R33, neuronas32, neuronas33  #Destruct objects
finaliza=np.size(Tasas, axis=1)
cols=headers["Data"]
#-------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------Cuerpo de los cálculos     ----------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
m=np.zeros((np.size(neuronas, axis=0), finaliza-cols+1))
m_A=np.zeros_like(m)
beta=np.zeros_like(m)
beta_A=np.zeros_like(m)
pm=np.zeros_like(m)
pbeta=np.zeros_like(m)
pRMSElineal=np.zeros_like(m)
pRMSEsigm=np.zeros_like(m)
pm_A=np.zeros_like(m)
pbeta_A=np.zeros_like(m)
pRMSElineal_A=np.zeros_like(m)
pRMSEsigm_A=np.zeros_like(m)
RMSEsigmoidal=np.zeros_like(m)
RMSEsigmoidalA=np.zeros_like(m)
RMSElinealA=np.zeros_like(m)
RMSElineal=np.zeros_like(m)
row=0
ti=time()
for serie, elec, unit  in neuronas:
    print("Neurona:     %f \t %f \t %f"%(serie, elec, unit))
    maskneu=(Tasas[:, headers["Serie"] ]==serie)*(Tasas[:, headers["Elec"] ]==elec)*(Tasas[:, headers["Uni"] ]==unit)
    submatriz=Tasas[maskneu,  :]  #Obtiene submatriz de datos
    maskTac=(submatriz[:, headers["Amp"]]==0) + (submatriz[:, headers["Amp"]]>3)
    maskAud=submatriz[:, headers["Amp"]]<3
    tmp=ALS.fitmatlab(submatriz[ (submatriz[:, headers["Amp"]]<3)*(submatriz[:, headers["Amp"]]>0), headers["Amp"] ])
    submatriz[(submatriz[:, headers["Amp"]]<3)*(submatriz[:, headers["Amp"]]>0), headers["Amp"]]=tmp
    #xampperTac=np.zeros((np.sum(maskTac), nperm))
    #xampperAud=np.zeros((np.sum(maskAud), nperm))
    #ALSexp.permarray(submatriz[maskTac, headers["Amp"]], xampperTac)  # genera  matriz  permutada
    #ALSexp.permarray(submatriz[maskAud, headers["Amp"]], xampperAud)  # genera la matriz  permutada
    yrateTac=np.zeros(np.sum(maskTac))
    yrateAud=np.zeros(np.sum(maskAud))
    colss=0    #Inicio de la variable que va sobre el tiempo
    model_lin=np.array([0, 0], dtype=np.float64)
    while(cols + colss< finaliza):     #Ciclo que va sobre el tiempo,
        yrateTac[:]=submatriz[maskTac, cols + colss]
        model_lin=ALS.linearfit((submatriz[maskTac, headers["Amp"]]).astype(np.float64), yrateTac.astype(np.float64),  np.sum(maskTac.astype(np.int32)))
        m[row, colss] =model_lin[0]
        linRMSE=np.sum((yrateTac-ALS.linear(submatriz[maskTac, headers["Amp"]], model_lin[0], model_lin[1]))**2 )
        modelsig, covsig=curve_fit(ALS.sigmoidal, xdata=submatriz[maskTac, headers["Amp"]], ydata=yrateTac, maxfev = 8000, p0=(5, 2, 1, 10), bounds=((0.1, -5, 0.1, 7), (50, 5, 30, 15)))
        sigmRMSE=np.sum((yrateTac-ALS.sigmoidal(submatriz[maskTac, headers["Amp"]], modelsig[0], modelsig[1], modelsig[2],  modelsig[3]))**2 )
        beta[row, colss]=modelsig[2]
        RMSEsigmoidal[row, colss]=sigmRMSE
        RMSElineal[row, colss]=linRMSE
        # Ahora acústico
        yrateAud[:]=submatriz[maskAud, cols + colss]
        model_lin=ALS.linearfit((submatriz[maskTac, headers["Amp"]]).astype(np.float64), yrateTac.astype(np.float64),  np.sum(maskTac.astype(np.int32)))
        m_A[row, colss]=model_lin[0]
        linRMSE=np.sum((yrateAud-ALS.linear(submatriz[maskAud, headers["Amp"]], model_lin[0], model_lin[1]))**2 )
        modelsig, covsig=curve_fit(ALS.sigmoidal, xdata=submatriz[maskAud, headers["Amp"]], ydata=yrateAud, maxfev = 8000, p0=(5, 2, 1, 10), bounds=((0.1, -5, 0.1, 7), (50, 5, 30, 15)))
        sigmRMSE=np.sum((yrateAud-ALS.sigmoidal(submatriz[maskAud, headers["Amp"]], modelsig[0], modelsig[1], modelsig[2],  modelsig[3]))**2 )
        beta_A[row, colss]=modelsig[2]
        RMSEsigmoidalA[row, colss]=sigmRMSE
        RMSElinealA[row, colss]=linRMSE
        xampTac=submatriz[maskTac, headers["Amp"]]
        xampAud=submatriz[maskAud, headers["Amp"]]
        #Permutaciones táctiles (paralelizada con ray)
        pm[row, colss], pbeta[row, colss], pRMSElineal[row, colss], pRMSEsigm[row, colss]=ALS.permutaciones(nperm, m[row, colss], beta[row, colss], yrateTac, xampTac,  RMSElineal[row, colss], RMSEsigmoidal[row, colss])
        # Repetición para el caso acústico (paralelizada con ray)
        pm_A[row, colss], pbeta_A[row, colss], pRMSElineal_A[row, colss], pRMSEsigm_A[row, colss]=ALS.permutaciones(nperm, m_A[row,colss], beta_A[row, colss], yrateAud, xampAud,  RMSElinealA[row, colss], RMSEsigmoidalA[row, colss])
        colss+=1
    row+=1
    break   # Corta para realizar la comparación
print("Tomó ", time()-ti, "de tiempo")


#Guardado de archivos
#a=open("Fit_comparison_Neur_Ind_TAcAud_Nperm-1000.pkl", "wb")
#results={"pm_A":pm_A, "pbeta_A":pbeta_A, "pRMSElineal_A": pRMSElineal_A, "pRMSEsigm_A": pRMSEsigm_A, "pm":pm, "pbeta":pbeta, "pRMSElineal": pRMSElineal, "pRMSEsigm": pRMSEsigm,   "beta_A" : beta_A, "RMSEsigmoidalA":RMSEsigmoidalA, "RMSElinealA": RMSElinealA,  "beta":beta,  "m":m, "m_A":m_A, "RMSEsigmoidal": RMSEsigmoidal, "RMSElineal": RMSElineal, "neuronas":neuronas, "cabeceras_neuronas":headers, "Python_Version":"3.8.6" }
#pickle.dump(results, a)
#a.close()
 
## Archivo json para compatibilidad con otros lenguajes
#results={"pm_A":tuple(pm_A), "pbeta_A":tuple(pbeta_A), "pRMSElineal_A": tuple(pRMSElineal_A), "pRMSEsigm_A": tuple(pRMSEsigm_A), "pm":tuple(pm), "pbeta":tuple(pbeta), "pRMSElineal": tuple(pRMSElineal), "pRMSEsigm": tuple(pRMSEsigm),   "beta_A" :tuple( beta_A), "RMSEsigmoidalA":tuple(RMSEsigmoidalA), "RMSElinealA": tuple(RMSElinealA),  "beta":tuple(beta),  "m":tuple(m), "m_A":tuple(m_A), "RMSEsigmoidal": tuple(RMSEsigmoidal), "RMSElineal": tuple(RMSElineal), "neuronas":tuple(neuronas), "cabeceras_neuronas":headers, "Python_Version":"3.8.6" }
#a=open("Fit_comparison_Neur_Ind_TAcAud_Nperm-1000.json", "w")
#json.dump(results, a)
#a.close()


