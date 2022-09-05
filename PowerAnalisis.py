from importlib import reload
import numpy as np
#import scipy.fft as fft
import scipy.signal as sn
import matplotlib.pyplot as plt
import lfpy as lfp
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import Phd_ext as Pext
reload(lfp)
#from Information import Mutual_information as minf   # Importa la función optimizada de información mutua.
#from skimage import measure
from numpy import log2
from numba import njit
import pickle as pkl
from os.path import lexists

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------

#                                                     DEFINE FUNCIONES IMPORTANTES

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
def sigmoidal(X, a, b, c, d):
    return (float(a-b))/(1.0 + np.exp((float(d)- X.astype(float))/float(c))) + float(b)

def fitmatlab(x):
    from numpy import log
    return 9.904*log(96.58*x+12.75) + 4.749

def loadLFP(area, monkey, serie, orden, lti=1.5, duracion=4.5, return_psych=False, computador=1,  **kwargs):
    """
    Esta función carga el archivo de lfp alineado al segmento de estimulación, 
    esta función está pensada para cargar los archivos alineados en un segmento específico.
    
    La función requiere de los siguientes argumentos de entrada:
    * area       : área registrada.
    * monkey     : Número de mono 32 o 33 valores aceptados.
    * serie      : Archivo de matlab.
    * orden      : Refiere usar el orden del set. 2 por ejemplo suele usarse en RR032 para el set de incertidumbre.
    * lti        : (opcional), lti=1.5 indica el tiempo anterior al inicio del estímulo principal 
    * duracion   : (opcional), default=4.5. Indica la duración total del segmento a tomar.
    * return_psych: (opcional) Devuelve los valores de la psicofísica así como las cabeceras. Estas últimas en forma de diccionario. El valor por defecto es False
    * computador: (opcional). Este parámetro es para la versatilidad entre las diferentes computadoras en que se utilizará este código:
       1:  Indica el lap Lenovo (valor por defecto).
       2: Indica el lap Hp 
       3: Indica el computador del laboratorio en la sesion de Windows
    
    
    Senales, mascaras=loadLFP(area="S1", monkey=33, serie=92, orden=2)
    
      **Ahora devuelve la información psicofísica:**
     
    Senales, mascaras, Psyc=loadLFP(area="S1", monkey=33, serie=92, orden=2, return_psych=True)
    """
    fsamp=2000                          # La frecuencia de muestreo es de 2khz
    longitud=np.int32(duracion*fsamp)   # Tiempo en núm de puntos
    nelec=7                             # Número de electrodos
    #widthtf=0.5                         # Ancho de las ventanas en el espectrograma
    
    if area=="S1":
        patharea="S1_izq_MAT"
    elif area=="A1":    
        patharea="A1_izq_MAT"
    else:
        raise ValueError("Hasta el momento solo se está trabajando con las áreas S1 y A1")

    if monkey==33:
        if computador==1: #Lap Lenovo
            path="/home/sparra/AENHA/LFP33/LFP_RR033/%s"%(patharea)
            pathcomp="/home/sparra/AENHA/Database_RR033"
        elif computador==2: # Lap Hp
            path="/run/media/sparra/AENHA/LFP33/%s"%(patharea)
            pathcomp="/run/media/sparra/AENHA/Database_RR033"          
        else:  #Lab
            path="D:\\LFP33\\%s"%(patharea)
            pathcomp="D:\\Database_RR033"      
        #namedir="RR033"
        if serie<100:
            name="RR0330%d_00%d"%(serie, orden)
        elif serie<10:
            name="RR03300%d_00%d"%(serie, orden)
        else:
            name="RR033%d_00%d"%(serie, orden)
    elif monkey==32:
        if computador==1: #Lnv
            path="/home/sparra/AENHA/LFP32/%s"%(patharea)
            pathcomp="/home/sparra/AENHA/BaseDatosKarlitosNatsushiRR032"
        elif computador==2:  #Hp
            path="/run/media/sparra/AENHA/LFP32/%s"%(patharea)
            pathcomp="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032"
        else:    #Lab
            path="D:\\LFP32\\%s"%(patharea)
            pathcomp="D:\\BaseDatosKarlitosNatsushiRR032" 
        #namedir="RR032"
        if serie<100:
            name="RR0320%d_00%d"%(serie, orden)
        elif serie<10:
            name="RR03200%d_00%d"%(serie, orden)
        else:
            name="RR032%d_00%d"%(serie, orden)

    else:
        raise ValueError("Valor incorrecto, solo es posible trabajar con los datos del mono 32 y 33")
    #------------------------------------------------------------------------------

    # Carga un archivo de prueba
    file="%s/%s"%(path, name)
    #file="/run/media/sparra/AENHA/LFP33/S1_izq_MAT/RR033075_001"
    if (monkey==32 or monkey==33) and area=="S1":
        datalfp=lfp.loadlfp(file)
        if len(datalfp)==0:
            print("Load file with loadlfp method failed .... \ntrying with loadnsx method...")
            datalfp, _=lfp.loadnsx(file)
    elif (monkey==32 or monkey==33) and area=="A1":
        datalfp, _=lfp.loadnsx(file)
    else:
        raise ValueError("Monkey or area not supported.")
    #------------------------------------------------------------------------------
    ntrials=len(datalfp)                # Número de ensayos en la sesión 
    #------------------------------------------------------------------------------

    if area=="S1":
        try:
            Psic=np.loadtxt("%s/Text_s/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter=",")            
        except:
            Psic=np.loadtxt("%s/Text_s/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter="\t")
        try:
            Tiempos=np.loadtxt("%s/Text_s/%s/%s_T.csv"%(pathcomp, name, name), delimiter="\t")
        except:
            Tiempos=np.loadtxt("%s/Text_s/%s/%s_T.csv"%(pathcomp, name, name), delimiter=",")
    else:
        Psic=np.loadtxt("%s/Text_sA1/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter=",")
        Tiempos=np.loadtxt("%s/Text_sA1/%s/%s_T.csv"%(pathcomp, name, name), delimiter=",")
    #Tiempos=np.loadtxt("/run/media/sparra/AENHA/Database_RR033/Text_s/RR033077_001/RR033077_001_T.csv",delimiter="\t")
    #------------------------------------------------------------------------------

    ## Cabeceras y constantes importantes para el script
    T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
    P={"Ntrial":0, "Nclase":1, "Answer":2, "CAnswer":3, "Hit?":4, "ATac1":5, "frec1":6, "AAud1":7, "ATac2":8, "frec2":9, "AAud2":10, "ATac3":11, "frec3":12, "AAud3":13, "RxKd":14, "RxKU":15, "RxPB":16, "BT":17, "TRW":18, "Set":19, "Ronda":22}
    Tac_psic=(0.796406, 0.037770, 1.495931, 10.667152)
    Ac_psic=(0.891063, 0, 1.952362, 34.102654)
    xtac=np.linspace(0, 28, 500)
    xac=np.linspace(0, 69, 500)

    Thresholding=np.zeros((2, 3), dtype=np.float32)
    Thresholding[0, 1]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.75))]

    Thresholding[1, 1]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.75))]
    # Ahora para el umbral 0.5
    Thresholding[0, 2]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.5))]

    Thresholding[1, 2]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.5))]

    #print("Los intervalos supraumbrales van por encima del 75%")
    Thresholding[0, 0]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.25))]

    Thresholding[1, 0]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.25))]
    #print("Los intervalos umbrales van por encima del 25% y menor al 75%")
    #print("Los intervalos subumbrales van por debajo del 25% ")

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # Parámetro del filtro pasabajas
    fcut=120                                  # Frecuencia de corte para el filtro pasabajas (La zona de transición es de aprox 30)
    tam=20                                    # Variable para fijar el orden (tam ciclos de la frecuencia de corte)
    orden=np.int32(tam/fcut*fsamp)            # Orden del filtro pasabajas
    Q=90                                      # Factor de calidad del filtro notch (Remueve el ruido de linea [60 hz])
    b=sn.firwin(orden, 2*fcut/fsamp)              # Coeficientes para el filtro pasa bajas
    Senales=np.zeros((ntrials, longitud, nelec))
    a_notch, b_notch,=sn.iirnotch(60/nyquist, Q)
    for trial in range(len(datalfp)):
        inicio=np.int32((Tiempos[trial, T["iE3"]]-lti)*fsamp)
        for elec in range(nelec):    
            # Filtado pasabajas
            senalF=datalfp[trial][inicio:inicio+longitud, elec]         
            nor=len(senalF)
            senalF=np.concatenate((np.flip(senalF), senalF, np.flip(senalF)) )  # Espejeado para minimizar efectos de frontera
            senalF=sn.filtfilt(b, 1, senalF)                                   # Filtra refleja y vuelve a filtrar (no distorsión de fase. Filtro pasa-bajos)
            senalF=sn.filtfilt(a_notch, b_notch, senalF)                                   # Filtra refleja y vuelve a filtrar (no distorsión de fase. Filtro notch)
            senalF=senalF[nor:-1-nor+1]  
            if len(senalF)!=longitud:
                raise ValueError("senalF does not have the length required,... %d"%(nor))
            Senales[trial, :, elec]=senalF       #Extrae el segmento de interés
    #   #Realiza rereferenciado tipo CAR (Common Average Rereferencing)
        Senales[trial, :, :]=Senales[trial, :, :] - np.repeat(np.mean(Senales[trial, :, :], axis=1).reshape(9000, 1), nelec, axis=1)
    masktac=Psic[:, P["AAud3"]]>0    
    Psic[masktac, P["AAud3"]]=fitmatlab(Psic[masktac, P["AAud3"]])
    #Ahora viene el ordenamiento de los ensayos táctiles. Amplitud de menor a mayor
    maskSupra_T=np.where(Psic[:, P["ATac3"]]>=Thresholding[0, 1])[0]
    #Ahora viene el ordenamiento de los ensayos acústicos. Amplitud de menor a mayor
    maskSupra_A=np.where(Psic[:, P["AAud3"]]>=Thresholding[1, 1])[0]
    maskzero=np.where((Psic[:, P["ATac3"]]==0)*(Psic[:, P["AAud3"]]==0))[0]
    #Ahora viene el ordenamiento de los ensayos táctiles. Subumbrales
    maskSub_T=np.where((Psic[:, P["ATac3"]]<=Thresholding[0, 0])*(Psic[:, P["ATac3"]]>0)  )[0]
    #Ahora viene el ordenamiento de los ensayos acústicos. Subumbrales
    maskSub_A=np.where( (Psic[:, P["AAud3"]]<=Thresholding[1, 0])*(Psic[:, P["AAud3"]]>0) )[0]
    #Ahora viene el ordenamiento de los ensayos táctiles. Umbrales
    maskUmb_T=np.where( (Psic[:, P["ATac3"]]>Thresholding[0, 0])*(Psic[:, P["ATac3"]]<Thresholding[0, 1])  )[0]
    #Ahora viene el ordenamiento de los ensayos acústicos. Umbrales
    maskUmb_A=np.where((Psic[:, P["AAud3"]]>Thresholding[1, 0])*(Psic[:, P["AAud3"]]<Thresholding[1, 1]))[0]    
    if return_psych:  # 
        return Senales, [maskSupra_A, maskSupra_T, maskUmb_A, maskUmb_T, maskSub_A, maskSub_T ,maskzero], [Psic, P]
    else:
        return Senales, [maskSupra_A, maskSupra_T, maskUmb_A, maskUmb_T, maskSub_A, maskSub_T ,maskzero]


def loadlfplimpio(area, monkey, serie, orden, electrode, computador=1,  **kwargs):
    """
    Esta función carga los archivos con los nesayos libres de artefactos, Esta función no devuelve rereferenciado el archivo
    LFP, es necesario que el usuario realice este procedimiento en base a los electrodos seleccionados para el análisis.
    la función requiere los siguientes entradas

    Parameters
    ----------
    area : Str, Refiere al área registrada S1 o A1 hasta el momento.
        DESCRIPTION.
    monkey : Int, Refiere al mono, 32 o 33 únicamente
        DESCRIPTION.
    serie : Int, refiere al número de registro (usualmente el día.)
        DESCRIPTION.
    orden : Int, refiere al orden en que se registró el set.
        DESCRIPTION.
    electrode : Int, refiere al electrodo a cargar
    computador : TYPE, optional
        DESCRIPTION. The default is 1.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    Data. Devuelve el archivo específico
    """
    if area=="S1":
        patharea="S1_Izq_txt"
    elif area=="A1":    
        patharea="A1_Izq_txt"
    else:
        raise ValueError("Hasta el momento solo se está trabajando con las áreas S1 y A1")
    if monkey==33:
        if computador==1: #Lap Lenovo
            path="/home/sparra/AENHA/LFP33/%s/"%(patharea)
        elif computador==2: # Lap Hp
            path="/run/media/sparra/AENHA/LFP33/%s/"%(patharea)
        else:  #Lab
            path="D:\\LFP33\\%s\\"%(patharea)
        #namedir="RR033"
        name="m33_s%d_o%d_e%d.csv"%(serie, orden, electrode)

    elif monkey==32:
        if computador==1: #Lnv
            path="/home/sparra/AENHA/LFP32/%s/"%(patharea)
        elif computador==2:  #Hp
            path="/run/media/sparra/AENHA/LFP32/%s/"%(patharea)
        else:    #Lab
            path="D:\\LFP32\\%s\\"%(patharea)
        name="m32_s%d_o%d_e%d.csv"%(serie, orden, electrode)
    else:
        raise ValueError("Valor incorrecto, solo es posible trabajar con los datos del mono 32 y 33")
    Data=np.loadtxt(path + name, delimiter=",", skiprows=(1))
    return Data
    
@njit("float32(double[:, :] , int32, int32, double[:])")
def Mutual_information(PrIs,  nrrn,  ncol,  Ps):
    """
    Program written by Sergio Parra S\'anchez. 
    This function calculates the mutual information of the 2 distributions:
    *2 PrIs  Probability of r given s have ocurred
    *3 Ps    Probability of the stimulus
    
    
    This function calculates the mutual information according to the following
    expression:
        I=\sum_{r} \sum_{s} {  P(s)*P(r|s)*log2(   P(r|s)/(P(r) )   }
    
    Pr must be an array with two columns and multiple rows
    PrIs  must be a matrix where the columns are sorted in the same
    order than Ps
    
    """
    I=0.0                     # Initialize the value of mutual information
    for stim in range(ncol):  # Sum over stimuli
        for r in range(nrrn): # Sum over any neural-response-metric outcome
            Pr=0
            for stim2 in range(ncol):
                 Pr+=PrIs[r, stim2]*Ps[stim2]  # Probabilidad marginal.
            if Pr>0 and PrIs[r, stim]>0:       # By convention zero develops zero
                 I+=Ps[stim]*PrIs[r, stim]*log2(PrIs[r, stim]/Pr)
    return I
    
@njit()
def infpermut(nperm, BinPhaseEdges, FasesT, nbinsPhase, PsT, AmpValuesT, permatrix, InfPhaseT):
    
    _, lent, nbands=np.shape(FasesT)
    for permi in range(nperm):
        for j in range(nbands-1):
            for ti in range(lent):
                distributionsPhaseT=np.zeros((nbinsPhase, len(PsT)), dtype=np.float64)
                i=0
                for amp in AmpValuesT:
                    masktmp=permatrix[:, permi]==amp
                    distributionsPhaseT[:, i], _  =np.histogram(FasesT[masktmp, ti, j], bins=BinPhaseEdges)
                    distributionsPhaseT[:, i]=distributionsPhaseT[:, i]/np.sum(masktmp)
                    i+=1
                # Ahora el cálculo de información
                InfPhaseT[j, ti, permi]=Mutual_information(distributionsPhaseT, nbinsPhase, len(PsT), PsT)
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(monkey, serie, orden, area, lti, fsamp, transicion, nbinsPhase, nbinsPow, pathdir, computador):
    """
    Función principal, esta función realiza el cálculo de la información de fase y potencia para las modalidades
    táctil y auditiva, este código asume que se trata del set de incertidumbre.
    
    Esta función tiene las siguientes parámetros de entrada:
    
    1.- monkey  : Id del mono a trabajar, 32 o 33.
    2.- serie   : Serie de registro, día de registro, entero. 187, 152 p. ej
    3.- orden   : Orden de registro, 1, 2, 3... 2 es el default de incertidumbre para el mono 32
    4.- area    . 
    5.- lti
    6.- transicion
    7.- nbinsPhase
    8.- nbinsPow
    9.- pathdir
    10.- computador
    
    """
    from time import time
    nyquist=fsamp/2
    tinicial=time()
    Senales, mascaras, Psych=loadLFP(area=area, monkey=monkey, serie=serie, orden=orden, return_psych=True, computador=computador)
    tfinal=time()
    print("La carga de archivos tomó: ", tfinal-tinicial)
    IdealResponse=(0, 0, 1, 1, 0, 0)    # Respuesta ideal del filtro. 
    ntrials, lent, nelec=Senales.shape  # Obtención del número de ensayos, longitud temporal, número de electrodos
    band=np.arange(4, 34, 4)           # bandas para el filtrado en minisegmentos
    band=np.concatenate( ([1], band, np.arange(40, 122, 8) ))
    ordenlabel=( "Acoustic Suprathreshold",
     "Tactile Suprathreshold",
     "Acoustic Threshold",
     "Tactile Threshold",
     "Acoustic Subthreshold",
     "Tactile Subthreshold",
     "No stimulus")

    amplitudes=Psych[0][:, Psych[1]["ATac3"]] + Psych[0][:, Psych[1]["AAud3"]]
    AmpValues, Nampl=np.unique(amplitudes, return_counts=True)
    # Constantes de los filtros
    filt0=np.zeros((len(band)), dtype=np.int32)
    filterweights=[]
    for j in range( len(band)-1):
        filt0[j]=4*(fsamp//band[j]) + 1
        freqs=[0, (1-transicion)*band[j], band[j],  band[j + 1], (1 + transicion)*band[j + 1], nyquist]
        filterweights.append(sn.firls(filt0[j], np.array(freqs)/nyquist, IdealResponse))
    tinicial=time()
    # Ahora el filtrado y extracción de fase y potencia con los parámetros previamente calculados    
    for elec_i in range(1): #nelec1
        # Guardar archivos con los cálculos de la fase por banda, por ensayo y por electrodo. Archivos csv. Separación por commas.
        namefilePhase="RR0%d%d_%d_%d-elec_InfPhase.csv"%(monkey, serie, orden, elec_i + 1) 
        filephase=open("%s/%s"%(pathdir, namefilePhase), "a+")   
        namefilePower="RR0%d%d_%d_%d-elec_InfPower.csv"%(monkey, serie, orden, elec_i + 1)
        filepower=open("%s/%s"%(pathdir, namefilePower), "a+")
        SenalesPhase=np.zeros((ntrials, lent, len(band)))    #Arreglo que mantiene en el último eje la banda de frecuencia
        SenalesAmp=np.zeros((ntrials, lent, len(band)))    #Arreglo que mantiene en el último eje la banda de frecuencia
        headers1="mono:%d serie:%d orden:%d electrodo:%d\n"%(monkey, serie, orden, elec_i + 1)
        filephase.write(headers1)
        filepower.write(headers1)
        SenalPower=[]
        SenalPhase=[]
        np.savetxt(filephase, np.reshape(band, (1, len(band))), delimiter=",", newline="\n")
        np.savetxt(filepower, np.reshape(band, (1, len(band))), delimiter=",", newline="\n")
        for j in range( len(band)-1):
            for tr in range(ntrials):  #ntrials
                snltmp=np.concatenate((np.flip(Senales[tr, :, elec_i]), Senales[tr, :, elec_i], np.flip(Senales[tr, :, elec_i])))
                snltmp=sn.filtfilt(b=filterweights[j], a=1, x=snltmp, padlen=filt0[j])
                snltmp=snltmp[lent:-1-lent+1]
                snltmp=sn.hilbert(snltmp)
                SenalesPhase[tr, :, j]=np.angle(snltmp)
                SenalesAmp[tr, :, j]=np.real((snltmp)*np.conj(snltmp) )
                #plt.plot(SenalesAmp[tr, :, j])
            print("Cálculo de las potencias y fases para el electrodo %d y banda: %d-%d Finalizado."%(elec_i+1, band[j], band[j+1]))              
        SenalPower.append(SenalesAmp)
        SenalPhase.append(SenalesPhase)
    tfinal=time()
    print("La demora del filtrado es: ", tfinal-tinicial)        
    # Continua la graficación de los espectrogramas con normalización porcentual (la misma que Saskia.)
    for elec_i in range(1):   #nelec
        reference=np.mean(SenalPower[elec_i][:,0: 600 ,:], axis=0)   # Media por ensayos
        reference=np.mean(reference, axis=0)                        # Media sobre el tiempo
        reference=np.repeat(reference.reshape(20, 1), 9000, axis=1)
        for masc_i in range(len(mascaras)):
            fig, ax=plt.subplots(1,1, figsize=(9, 9))
            fig.suptitle(ordenlabel[masc_i], fontsize=20)
            tmp=np.transpose(np.mean(SenalPower[elec_i][mascaras[masc_i], :, :], axis=0))
            #tmp=((tmp-reference)/reference)*100
            tmp=10*np.log10(tmp/reference)
            im=ax.imshow(tmp, origin="lower", aspect="auto", extent=[-1.5, 3, 1, len(band) +1], cmap="seismic", vmin=-10, vmax=10)
            ax.plot((0, 0, 0.5, 0.5), (1, len(band)+1, len(band)+1, 1), c="black", ls="--", lw=3)
            ax.set_xlabel("Time [s]", fontsize=20)
            ax.set_ylabel("Frequency [Hz]", fontsize=20)
            ax.tick_params(labelsize=18)
            ax.set_yticks(np.arange(1, len(band) +1, 3))
            ax.set_yticklabels(str(i) for i in band[0::3] )
            ax.set_ylim([1, 18])
            # Create divider for existing axes instance
            divider3 = make_axes_locatable(ax)
            # Append axes to the right of ax3, with 20% width of ax3
            cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            cbar3 = plt.colorbar(im, cax=cax3)
            for t in cbar3.ax.get_yticklabels():
                t.set_fontsize(18)
            cbar3.set_label("Power Change[%]", fontsize=20)
    return None

def Main(monkey, serie, orden, area, electrodes, fsamp, transicion, pathdir, computador, Norm=0, mod=0,  **kwargs):
    """
    Función principal, esta función realiza el cálculo de la fase y potencia para las modalidades. 
    Esta función guarda los archivos de fase y potencia en formato pickle, usa para ello los archivos ubicados
    en la carpeta de limpieza, con ello se asume que todos los electrodos tienen los mismos ensayos así como que 
    los primeros 23 columnas refieren a la psicofísica, esto es, esta función toma los archivos producidos por el script
    de LimpiaLFP.py ubicado en este mismo directorio. Esta función permite guardar las figuras producidas también. Este guardado es automático
    y se realiza en la carpeta pathdir
    
    
    Esta función tiene las siguientes parámetros de entrada:
    
    1.- monkey  : Id del mono a trabajar, 32 o 33.
    2.- serie   : Serie de registro, día de registro, entero. 187, 152 p. ej
    3.- orden   : Orden de registro, 1, 2, 3... 2 es el default de incertidumbre para el mono 32
    4.- area    . 
    5.- electrodes
    6.- transicion
    7.- pathdir
    8.- computador
    9.- Norm.-  Refiere al tipo de normalización: 1.- Z-score, 2.- Porcentual  Otro.-Decibeles (valor por defecto), 
    10.. mod.-Refiere a la normalización, 1 realizará normalización al promedio, otro (default), realizará una normalización por ensayo y mostrará el promedio.
    
    """
    if Norm==2:   # Porcentual change
        @njit()
        def normalization(matrix, row, cols, ntrials):
            for trial_i in range(ntrials):
                for band_i in range(row):
                    reference=np.mean(matrix[band_i, 0: 600, trial_i]) + 1e-6   # Referencia para la normalización                
                    matrix[band_i, :, trial_i]=((matrix[band_i, :, trial_i]-reference)/reference)*100
            return None
        lab="%"
    elif Norm==1:    #Zscore change 
        @njit()
        def normalization(matrix, row, cols, ntrials):
            for trial_i in range(ntrials):
                for band_i in range(row):
                    media=np.mean(matrix[band_i, 0: 600, trial_i])   # media para la normalización   
                    sigma=np.std(matrix[band_i, 0: 600, trial_i]) + 1e-6   # sigma para la normalización                
                    matrix[band_i, :, trial_i]=(matrix[band_i, :, trial_i]-media)/sigma
            return None
        lab="$\sigma$"
    else:  #Decibeles change
        @njit()
        def normalization(matrix, row, cols, ntrials):
            modificada=np.zeros((row, cols, ntrials), np.float32)
            for trial_i in range(ntrials):
                for band_i in range(row):
                    reference=np.mean(matrix[band_i, 0: 600, trial_i]) + 1e-6   # Referencia para la normalización   
                    modificada[band_i, :, trial_i]=10*np.log10((matrix[band_i, :, trial_i] + 1e-6)/reference)
            return modificada
        lab="db"
    originalpath=os.getcwd()
    if monkey==33:
        if computador==1:
            alternativepath="/home/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
        elif computador==2:
            alternativepath="/run/media/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
        elif computador==3:
            alternativepath="D:/LFP33/FaseyPotenciaS1RR033"
        else:
            raise ValueError("Computador inválido")
    elif monkey==32:
        if computador==1:
            alternativepath="/home/sparra/AENHA/LFP32/FaseyPotenciaS1RR032"
        elif computador==2:
            alternativepath="/run/media/sparra/AENHA/LFP32/FaseyPotenciaS1RR032"
        elif computador==3:
            alternativepath="D:/LFP32/FaseyPotenciaS1RR032"
        else:
            raise ValueError("Computador inválido")
    else:
        raise ValueError("Datos desconocidos")
    nyquist=fsamp/2
    ind=0
    for elec_i in electrodes:
        if ind==0:
            tmp=loadlfplimpio(area=area, monkey=monkey, serie=serie, orden=orden, electrode=elec_i, computador=computador)
            Senales=np.zeros((np.size(tmp, axis=0), np.size(tmp, axis=1), len(electrodes)), dtype=np.float32)
            Senales[:, :, ind]=np.copy(tmp)
            ntrials, lent=tmp.shape
            lent-=23   # Son 23 las columnas extra por la psicofisica
            ind+=1
        else:
            Senales[:, :, ind]=loadlfplimpio(area=area, monkey=monkey, serie=serie, orden=orden, electrode=elec_i, computador=computador)
            ind+=1
    # Ahora viene la etapa de common average rereferencing
    for trial_i in range(ntrials):
        Senales[trial_i, 23::, :]=Senales[trial_i, 23::, :]- np.repeat(np.mean(Senales[trial_i, 23::, :], axis=1).reshape(9000, 1), len(electrodes), axis=1)
    # Finaliza la referencia CAR
    IdealResponse=(0, 0, 1, 1, 0, 0)    # Respuesta ideal del filtro. 
    band=np.arange(4, 34, 4)            # bandas para el filtrado en minisegmentos
    band=np.concatenate( ([1], band, np.arange(40, 122, 8) ))
    # Cabeceras y constantes importantes para la función
    P={"Ntrial":0, "Nclase":1, "Answer":2, "CAnswer":3, "Hit?":4, "ATac1":5, "frec1":6, "AAud1":7, "ATac2":8, "frec2":9, "AAud2":10, "ATac3":11, "frec3":12, "AAud3":13, "RxKd":14, "RxKU":15, "RxPB":16, "BT":17, "TRW":18, "Set":19, "Ronda":22}
    Tac_psic=(0.796406, 0.037770, 1.495931, 10.667152)
    Ac_psic=(0.891063, 0, 1.952362, 34.102654)
    xtac=np.linspace(0, 28, 500)
    xac=np.linspace(0, 69, 500)

    Thresholding=np.zeros((2, 3), dtype=np.float32)
    Thresholding[0, 1]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.75))]
    Thresholding[1, 1]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.75))]
    # Ahora para el umbral 0.5
    Thresholding[0, 2]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.5))]
    Thresholding[1, 2]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.5))]
    #print("Los intervalos supraumbrales van por encima del 75%")
    Thresholding[0, 0]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.25))]
    Thresholding[1, 0]=xac[np.argmin(np.abs(sigmoidal(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.25))]
    mascaras=[np.where(Senales[:, P["AAud3"], 0]>=Thresholding[1, 1])[0],
              np.where(Senales[:, P["ATac3"], 0]>=Thresholding[0, 1])[0],
              np.where((Senales[:, P["AAud3"], 0]>Thresholding[1, 0])*(Senales[:, P["AAud3"], 0]<Thresholding[1, 1]))[0],
              np.where((Senales[:, P["ATac3"], 0]>Thresholding[0, 0])*(Senales[:, P["ATac3"], 0]<Thresholding[0, 1])  )[0],
              np.where((Senales[:, P["AAud3"], 0]<=Thresholding[1, 0])*(Senales[:, P["AAud3"], 0]>0)  )[0],
              np.where((Senales[:, P["ATac3"], 0]<=Thresholding[0, 0])*(Senales[:, P["ATac3"], 0]>0)  )[0],
              np.where((Senales[:, P["ATac3"], 0]==0)*(Senales[:, P["AAud3"], 0]==0))[0] ]
    ordenlabel=( "Acoustic Suprathreshold",
     "Tactile Suprathreshold",
     "Acoustic Threshold",
     "Tactile Threshold",
     "Acoustic Subthreshold",
     "Tactile Subthreshold",
     "No stimulus")
    tmpFlag=True
    # Constantes de los filtros
    if lexists("Filterparameters.pkl"):
        ftmp=open("Filterparameters.pkl", "rb")
        fcoef=pkl.load(ftmp)
        comptmp=fcoef["Bandas"]==band
        ftmp.close()
        if comptmp.all(): # No es necesario recalcular coefiecientes
            filt0=fcoef["Orden"]
            filterweights=fcoef["Coeficientes"]
            tmpFlag=False
            del ftmp, fcoef, comptmp      # Borra variables temporales e innecesarias para el futuro.
    # Si los coeficientes no han sido precalculados, entonces es necesario recalcular.
    if tmpFlag:
        filt0=np.zeros((len(band)), dtype=np.int32)
        filterweights=[]
        for j in range( len(band)-1):
            filt0[j]=4*(fsamp//band[j]) + 1
            freqs=[0, (1-transicion)*band[j], band[j],  band[j + 1], (1 + transicion)*band[j + 1], nyquist]
            filterweights.append(sn.firls(filt0[j], np.array(freqs)/nyquist, IdealResponse))
    # Ahora el filtrado y extracción de fase y potencia con los parámetros previamente calculados    
    for elec_i in range(len(electrodes)): #nelec1
        # Guardar archivos con los cálculos de la fase por banda, por ensayo y por electrodo. Archivos csv. Separación por commas.
        SenalesPhase=np.zeros((len(band), lent, ntrials))    #Arreglo que mantiene en el último eje la banda de frecuencia
        SenalesAmp=np.zeros((len(band), lent, ntrials))    #Arreglo que mantiene en el último eje la banda de frecuencia
        for j in range( len(band)-1):
            for tr in range(ntrials):  #ntrials
                snltmp=np.concatenate((np.flip(Senales[tr, 23::, elec_i]), Senales[tr, 23::, elec_i], np.flip(Senales[tr, 23::, elec_i])))
                snltmp=sn.filtfilt(b=filterweights[j], a=1, x=snltmp, padlen=filt0[j])
                snltmp=snltmp[lent:-1-lent+1]
                snltmp=sn.hilbert(snltmp)
                SenalesPhase[j, :, tr]=np.angle(snltmp)
                SenalesAmp[j, :, tr]=np.real((snltmp)*np.conj(snltmp) )
            print("Cálculo de las potencias y fases para el electrodo %d y banda: %d-%d Finalizado."%(electrodes[elec_i], band[j], band[j+1]))              
        ## Guardado de archivo de fase y potencia
        os.chdir(alternativepath)
        f=open("m%d_s%d_o%d_e%d_Spectrum.pkl"%(monkey, serie, orden, electrodes[elec_i]), "wb")
        datadict={"Inf":{"Monkey":monkey, "serie":serie, "orden":orden, "electrode":electrodes[elec_i],
                         "method":"Filter-Hilbert", "Rereferencing":True, "Area":area, "Comments": "Data was filtered by firls method with a transition range of %.3f"%(transicion*100)},
                  "banda":band,"Power":SenalesAmp, "Phase": SenalesPhase, "Psychophysics":Senales[:, 0:23, 0]}        
        pkl.dump(datadict, f)
        f.close()
        os.chdir(originalpath)
       # Continua la graficación de los espectrogramas con normalización 
        if mod==1:   # Normalización por promedio       
            reference=np.mean(SenalesAmp[:,0: 600 ,:], axis=2)             # Media por ensayos
            reference=np.mean(reference, axis=1)                           # Media sobre el tiempo
            reference=np.repeat(reference.reshape(20, 1), 9000, axis=1)
            reference=reference+1e-6
            if Norm==1: #Zscore 
                referencesigma=np.mean(SenalesAmp[:, 0: 600 ,:], axis=2)   # Media por ensayos
                referencesigma=np.std(referencesigma, axis=1)              # std sobre el tiempo
                referencesigma=np.repeat(referencesigma.reshape(20, 1), 9000, axis=1)
            for masc_i in range(len(mascaras)):
                tmp=np.mean(SenalesAmp[:, :, mascaras[masc_i]], axis=2)+1e-6
                fig, ax=plt.subplots(1,1, figsize=(9, 9))
                fig.suptitle(ordenlabel[masc_i], fontsize=20)
                if Norm==2:   # Porcentual change
                    tmp=((tmp-reference)/reference)*100
                    lab="%"
                elif Norm==1:    #Zscore change 
                    tmp=((tmp-reference)/referencesigma)
                    lab="$\sigma$"
                else:  #Decibeles change
                    tmp=10*np.log10(tmp/reference)
                    lab="db"
                im=ax.imshow(tmp, origin="lower", aspect="auto", extent=[-1.5, 3, 1, len(band) +1], cmap="seismic", vmin=-10, vmax=10)
                ax.plot((0, 0, 0.5, 0.5), (1, len(band)+1, len(band)+1, 1), c="black", ls="--", lw=3)
                ax.set_xlabel("Time [s]", fontsize=20)
                ax.set_ylabel("Frequency [Hz]", fontsize=20)
                ax.tick_params(labelsize=18)
                ax.set_yticks(np.arange(1, len(band) +1, 3))
                ax.set_yticklabels(str(i) for i in band[0::3] )
                ax.set_ylim([1, 18])
                # Create divider for existing axes instance
                divider3 = make_axes_locatable(ax)
                # Append axes to the right of ax3, with 20% width of ax3
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                cbar3 = plt.colorbar(im, cax=cax3)
                for t in cbar3.ax.get_yticklabels():
                    t.set_fontsize(18)
                cbar3.set_label("Power Change[%s]"%(lab), fontsize=20)

        else:  # Normalización por ensayo y luego promedio, minimiza outliers, necesaria cuando son pocos ensayos.
            SenalesAmp=normalization(SenalesAmp,  len(band), np.size(SenalesAmp, axis=1), ntrials)
            for masc_i in range(len(mascaras)):
                fig, ax=plt.subplots(1,1, figsize=(9, 9))
                fig.suptitle(ordenlabel[masc_i], fontsize=20)
                im=ax.imshow(np.mean(SenalesAmp[:, :, mascaras[masc_i]], axis=2), origin="lower", aspect="auto", 
                             extent=[-1.5, 3, 1, len(band) +1], cmap="seismic", vmin=-10, vmax=10)
                ax.plot((0, 0, 0.5, 0.5), (1, len(band)+1, len(band)+1, 1), c="black", ls="--", lw=3)
                ax.set_xlabel("Time [s]", fontsize=20)
                ax.set_ylabel("Frequency [Hz]", fontsize=20)
                ax.tick_params(labelsize=18)
                ax.set_yticks(np.arange(1, len(band) +1, 3))
                ax.set_yticklabels(str(i) for i in band[0::3] )
                ax.set_ylim([1, 18])
                # Create divider for existing axes instance
                divider3 = make_axes_locatable(ax)
                # Append axes to the right of ax3, with 20% width of ax3
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                cbar3 = plt.colorbar(im, cax=cax3)
                for t in cbar3.ax.get_yticklabels():
                    t.set_fontsize(18)
                cbar3.set_label("Power Change[%s]"%(lab), fontsize=20)
            # Guardado de la figura
        if len(pathdir)>0:
            os.chdir(pathdir)
            fig.savefig("m%d_s%d_o%d_e%d_%s.svg"%(monkey, serie, orden, electrodes[elec_i], ordenlabel[masc_i]))
            os.chdir(originalpath)
        plt.close("all")
    return None


#monkey=33                           # Mon, solo 32 o 33
#serie=80                            # Serie o día de registro
#orden=3                             # Orden del set. 2 es Unc en RR032
#area="S1"                           # Area de registro
#lti=1.5                             # Tiempo de corte hacia atrás
#nyquist=1000                        # Frecuencia de Nyquist
#fsamp=2000                          # Frecuencia de muestreo 
#transicion=0.05                     # Ventana de transición (porcentual/100)
#nbinsPhase=4                        # Número de bines para el cálculo de información de fase
#nbinsPow=8                          # Número de bines para el cálculo de información de energía o potencia
#computador=1

#if computador==1:
#    pathdir="/home/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
#elif computador==2:
#    pathdir="/run/media/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
#else:
#    pathdir="D:\\LFP33\\FaseyPotenciaS1RR033"

#main(monkey, serie, orden, area, lti, fsamp, transicion, nbinsPhase, nbinsPow, pathdir, computador)

#monkey=33                           # Mon, solo 32 o 33
#serie=(77,)                         # Serie o día de registro
#orden=(1,)                          # Orden del set. 2 es Unc en RR032
#electrodes=(1,2,3,4,5,6,7)          # Electrodos válidos en el estudio.
area="S1"                            # Area de registro
lti=1.5                              # Tiempo de corte hacia atrás
nyquist=1000                         # Frecuencia de Nyquist
fsamp=2000                           # Frecuencia de muestreo 
transicion=0.05                      # Ventana de transición (porcentual/100)
nbinsPhase=4                         # Número de bines para el cálculo de información de fase
nbinsPow=8                           # Número de bines para el cálculo de información de energía o potencia
computador=3

if computador==1:
    pathdir="/home/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
elif computador==2:
    pathdir="/run/media/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
else:
    pathdir="D:\\LFP33\\FaseyPotenciaS1RR033"

#main(monkey, serie, orden, area, lti, fsamp, transicion, nbinsPhase, nbinsPow, pathdir, computador)
      
#serie=[154, 155, 157, 158, 160, 162,  164,  166]
#orden=[2,    2,   2,   2,   2,   2,    2,    2 ]
class InputParameters:
    def __init__(self, monkey, serie, orden, electrodes):
        self.m=monkey
        self.s=serie
        self.o=orden
        self.e=electrodes
        return None

Parametros=[#InputParameters(33, 78, 3, (1, 2, 4, 5, 6, 7)),
            #InputParameters(33, 80, 3, (1, 2, 4, 5, 6, 7)), 
            #InputParameters(33, 82, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 83, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 84, 1, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 85, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 86, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 87, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 88, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 89, 3, (1, 2, 3, 4, 5, 6, 7)),
            ##InputParameters(33, 90, 3, (1, 2, 3, 4, 5, 6, 7)),
            #InputParameters(33, 91, 2, (1, 2, 3, 4, 5, 6, 7)),
            ##InputParameters(33, 92, 3, (2, 3, 4, 5, 6, 7)),
            InputParameters(33, 93, 3, (1, 3, 6, 7)),
            #InputParameters(33, 94, 1, (1, 2, 3, 6, 7)),   #Este está limpio (falta agregar),
            ] 
for i in range(len(Parametros)):
    Main(Parametros[i].m, Parametros[i].s, Parametros[i].o,
         area, Parametros[i].e, 2000, transicion,  pathdir, computador, Norm=0)
    
