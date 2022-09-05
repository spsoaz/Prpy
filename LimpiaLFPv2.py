"""

@author: sparra

Función de limpieza, esta función busca seleccionar los ensayos que debieran removerse  de los LFP. 
La primera forma de limpieza consiste en verificar los ensayos que no se mantuvieron en ninguna de las espigas recolectadas en el electrodo.

No obstante en cada caso se verá si el ensayo contiene elementos atípicos persistentes en el tiempo.
Este script ir\'a evolucionando hasta volverse el definitivo (con otro nombre) para
la remociÃ³n de artefactos en la seÃ±al del lfp mediante la tÃ©cnica de anÃ¡lisis de componentes
independientes.



Este archivo muestra una combinaci\'on as\'i como una extensi\'on de los archivos LimpiaLFP.py y ICA_pruebas.py 


"""
from importlib import reload
from numba import njit
import numpy as np
from lfpy import emd 
import scipy.signal as sn
import scipy.fft as fft
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
import lfpy as lfp
from os.path import join, lexists
from os import mkdir, getcwd, chdir
reload(lfp)
import matplotlib
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
#from os.path import join
np.random.seed(0)
import seaborn as sns
from sklearn.decomposition import FastICA, PCA
sns.set(rc={'figure.figsize':(11.7,8.27)})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
##                           Parametros del script para la visualizacion de ensayos con artefactos vs. sin ellos

monkey=32                           # Datos del mono a analizar
serie=170                           # Serie de registro (nombre original del mat file)
orden=2                             # Orden del set en ese día de registro. 
electrodes=(1, 2, 3, 4, 5, 6, 7)    # Electrodos que se van a cargar
area="S1"                           # Area de registro
lti=1.5                             # Tiempo de corte hacia atrás
computador=1                        # Computador en el que se harán los cálculos: 1: lnv, 2: hp, 3:lab
rmtrials=False                      # Para remover los ensayos identificados con artefactos. Por defecto falso


#serie=[75, 77, 78, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94] # Serie o día de registro
#orden=[1,   1,  3,  3,  3,  3,  1,  3,  3,  3,  3,  3,  3,  2,  3,  3,  1] # Orden  # Orden del set. 2 es Unc en RR032
#serie=[153, 154, 155, 157, 158, 160, 162, 164, 166]
# Estos sí se pudieron limpiar : 154, 155, 157, 158, 160, 162, 163, 164, 165, 166, 167, 170
#orden=[2,    2,   2,    2,   2,   2,  2,   2,   2]

if monkey==32:
    if computador==1:
        savename="/home/sparra/AENHA/LFP32/S1_Izq_txt"
    elif computador==2:
        savename="/run/media/sparra/AENHA/LFP32/S1_Izq_txt"
    else:
        savename="D:\\LFP32\\S1_Izq_txt"   
elif monkey==33:
    if computador==1:
        savename="/home/sparra/AENHA/LFP33/S1_Izq_txt"
    elif computador==2:
        savename="/run/media/sparra/AENHA/LFP33/S1_Izq_txt"
    else:
        savename="D:\\LFP33\\S1_Izq_txt" 
else:
    ValueError("Datos desconocidos, no se reconoce ese mono.")
        
    
    
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - 

class InputParameters:
    def __init__(self, monkey, serie, orden, electrodes, computador, lti):
        self.m=monkey
        self.s=serie
        self.o=orden
        self.e=electrodes
        self.lti=lti
        self.c=computador
        return None

# Define funciones importantes
@njit("double[:](double[:], double, double, double, double)")
def sigmoidal(X, a, b, c, d):
    return (a-b)/(1.0 + np.exp((d - X)/c)) + b

def fitmatlab(x):
    return 9.904*np.log(96.58*x+12.75) + 4.749

def findabsent(array):
    """
    Esta función buscará los elementos ausentes en un array que debe incrementarse uno a uno
    """
    diferencias=np.diff(array)
    if np.max(diferencias)==1:
        return ()   # Regresa una tupla vacía.
    else:           #Hay elementos ausentes
        faltantes=[]
        posiciones=np.where(diferencias>1)[0]
        for nope in range(len(posiciones)):
            faltantes.append(np.arange(array[posiciones[nope]]+1, array[posiciones[nope] +1] ))
        faltantes=np.hstack(faltantes)
    return faltantes

def Fourier(Senales, fsamp=2000, showfig=False):
    """
        Esta funci\'on devuelve las transformadas de Fourier del nd array Senales, el cual tiene la siguiente estructura
       
        Esta función adem\'as realiza la graficaci\'on de las transformadas  En cualquier caso es capaz de 
        
    Parameters
    ----------
    Senales : 3nd array
        Es la matriz de señales que se transformarán de acuerdo a Fourier
        dimensión 1: Son los ensayos
        dimension2 : son los tiempos
        dimension3: son los electrodos
        Senales es pues un arreglo de ensayosXTiempoXelectrodos
    fsamp: Escalar
        Refiere a la frecuencia de muestreo, el valor por defecto es 2000Hz
      Returns
    -------
    Transformada de Fourier .

    """
    ntrials, N, nelec=Senales.shape
    freqs=np.linspace(0, fsamp/2, N//2+1)
    SenalesX=np.zeros((ntrials,  N//2+1, nelec), dtype=np.float32)
    for trial_i in range(ntrials):
        for elec_i in range(nelec):
            SenalesX[trial_i, :, elec_i]=np.abs(2*fft.rfft(Senales[trial_i, :, elec_i]-np.mean(Senales[trial_i, :, elec_i]))/N)**2
                
    ## AHora la graficación
    if showfig:
        basal=np.zeros((nelec), dtype=np.float32)
        for elec_i in range(nelec):
            fig_i, ax_i=plt.subplots(1, 1, figsize=(9, 6))
            for trial_i in range(ntrials):
                ax_i.plot(freqs, np.log10(SenalesX[trial_i, :, elec_i]) + basal[elec_i], c="black")
                #basal[elec_i]+=np.max(np.abs(np.log10(SenalesX[trial_i, :, elec_i])))*1.1
            ax_i.spines["top"].set_visible(False)
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["left"].set_visible(False)
            ax_i.tick_params(labelsize=16)
            ax_i.set_xlabel("Frequencies [Hz]", fontsize=16)
            ax_i.set_xlim([0, 80])
            ax_i.set_title("Electrode: " + str(elec_i+1))
            fig_i.tight_layout()
        plt.show()
    return SenalesX
    


def loadLFP(area, monkey, serie, orden, lti=1.5, duracion=4.5, computador=1, **kwargs):
    """
    Esta función nos devuelve las señales de LFP recortadas, alineadas y filtradas con un
    filtro pasabajas con frecuencia de corte en 120. Esta función filtra en 60 Hz. 
    
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
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    ## Cabeceras y constantes importantes para la función
    T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
    P={"Ntrial":0, "Nclase":1, "Answer":2, "CAnswer":3, "Hit?":4, "ATac1":5, "frec1":6, "AAud1":7, "ATac2":8, "frec2":9, "AAud2":10, "ATac3":11, "frec3":12, "AAud3":13, "RxKd":14, "RxKU":15, "RxPB":16, "BT":17, "TRW":18, "Set":19, "Ronda":22}

    #print("Los intervalos umbrales van por encima del 25% y menor al 75%")
    #print("Los intervalos subumbrales van por debajo del 25% ")

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    ## Empecemos por analizar los lfp de los ensayos subumbrales

    # Parámetro del filtro pasabajas
    fcut=120                                  # Frecuencia de corte para el filtro pasabajas (La zona de transición es de aprox 30)
    tam=8                                     # Variable para fijar el orden (tam ciclos de la frecuencia de corte)
    orden=np.int32(tam/fcut*fsamp)            # Orden del filtro
    b=sn.firwin(orden, 2*fcut/fsamp)          # coeficientes del filtro   
    nyquist=1000                              # Frecuencia de Nyquist
    Q=90                                      # Factor de calidad del filtro notch (Remueve el ruido de linea [60 hz])
    a_notch, b_notch,=sn.iirnotch(60/nyquist, Q)
    Senales=np.zeros((ntrials, longitud, nelec))
    for trial in range(len(datalfp)):
        inicio=np.int32((Tiempos[trial, T["iE3"]]-lti)*fsamp)
        for elec in range(nelec):    
            # Filtado pasabajas
            senalF=datalfp[trial][inicio:inicio+longitud, elec]         
            nor=len(senalF)
            senalF=np.concatenate((np.flip(senalF),senalF, np.flip(senalF)) )  # Espejeado para minimizar efectos de frontera
            senalF=sn.filtfilt(b, 1, senalF)                                   # Filtra refleja y vuelve a filtrar (no distorsión de fase.)
            senalF=sn.filtfilt(a_notch, b_notch, senalF)                       # Filtra refleja y vuelve a filtrar (no distorsión de fase. Filtro notch)
            senalF=senalF[nor:-1-nor+1]  
            if len(senalF)!=longitud:
                raise ValueError("senalF does not have the length required,... %d"%(nor))
            Senales[trial, :, elec]=senalF       #Extrae el segmento de interés

    masktac=Psic[:, P["AAud3"]]>0    
    Psic[masktac, P["AAud3"]]=fitmatlab(Psic[masktac, P["AAud3"]])
    #
    Psic_sort=np.lexsort((Psic[:, 11], Psic[:,13]))
    return Senales, Psic_sort, Psic

#----------------------------------------------------------------------------

def plotoriginals(monkey, serie, orden, area, elecs, computador, lti=1.5, defaultntrials=150, rmtrials=False, **kwargs):
     """
       Programa para el análisis de tiempo frecuencia de las señales LFP.

       No se realizó un filtrado de ruido de línea porque posiblemente ya venga prefiltrada
       Solo hay un filtro pasabajas con frecuencia a 120 Hz. Hay una señal interesante a 105 o 75. Ver
       Hace uso de rereferenciado mediante la técnica de common average.
     
       Esta función genera los ERP de clasificando por grupos de acuerod a si son subumbrales, umbrales y supraumbrales.
     """
     originalpath=getcwd()
     cabecera="Ntrial, Nclase, Answer, CAnswer, Hit?, ATac1, frec1, AAud1, ATac2, frec2, AAud2, ATac3, frec3, AAud3, RxKd, RxKU, RxPB, BT, TRW, Set, Ronda, Señal"
     Senales, PsicSort, Psic=loadLFP(area, monkey, serie, orden, computador=computador)             # Carga de archivos
     if monkey==33:
         if computador==1 or computador==2:
             SpikesLists=np.loadtxt("/home/sparra/MEGA/Listas/S1_Inc_RR033.csv", delimiter=",", skiprows=(1))
             SpikesLists=SpikesLists[np.where(SpikesLists[:, 0]==serie)[0], :]
             SpikesLists=SpikesLists[:, (0, 2, 3, 4)]
         elif computador==3:
             SpikesLists=np.loadtxt("C:/Users/sparra/Documents/Listas/S1_Inc_RR033.csv", delimiter=",", skiprows=(1))
             SpikesLists=SpikesLists[np.where(SpikesLists[:, 0]==serie)[0], :]
             SpikesLists=SpikesLists[:, (0, 2, 3, 4)] 
         else:
             raise ValueError("No es un computador con direcciones conocidas")   
         tmp="LFP33"           
     elif monkey==32:
            tmp="LFP32"
            if computador==1 or computador==2:
                SpikesLists=np.loadtxt("/home/sparra/MEGA/Listas/S1_Inc_RR032.csv", delimiter=",", skiprows=(1))
                SpikesLists=SpikesLists[np.where(SpikesLists[:, 0]==serie)[0], :]
                SpikesLists=SpikesLists[:, (0, 2, 3, 4)]
            elif computador==3:
                SpikesLists=np.loadtxt("C:/Users/sparra/Documents/Listas/S1_Inc_RR032.csv", delimiter=",", skiprows=(1))
                SpikesLists=SpikesLists[np.where(SpikesLists[:, 0]==serie)[0], :]
                SpikesLists=SpikesLists[:, (0, 2, 3, 4)] 
            else:
                raise ValueError("No es un computador con direcciones conocidas") 
     else:
           raise ValueError("Datos del mono no soportados")
     if computador==1:
         filename="/home/sparra/AENHA/%s/ERP_results/Referenciados"%(tmp)
         if monkey==33:
             filespikes="/home/sparra/AENHA/Database_RR033/Text_s/RR0330%d_00%d"%(serie, orden)
         elif monkey==32:
             filespikes="/home/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s/RR032%d_00%d"%(serie, orden)
     elif computador==2:        
         filename="/run/media/sparra/AENHA/%s/ERP_results/Referenciados"%(tmp) 
         if monkey==33:
             filespikes="/run/media/sparra/AENHA/Database_RR033/Text_s/RR0330%d_00%d"%(serie, orden)
         elif monkey==32:
             filespikes="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s/RR032%d_00%d"%(serie, orden)
     elif computador==3:
         filename="D:\\%s\\ERP_results\\Referenciados"%(tmp)
         if monkey==33:
             filespikes="D:/Database_RR033/Text_s/RR0330%d_00%d"%(serie, orden)
         elif monkey==32:
             filespikes="D:/BaseDatosKarlitosNatsushiRR032/Text_s/RR032%d_00%d"%(serie, orden)
     ntrials, _, nelec=Senales.shape                                         # Número de electrodos 
     print("El número de electrodos son: ",nelec)
     failtrials=[]
     # Etapa de referenciado common average
     #for i in range(ntrials):
     #    Senales[i, :, :]=Senales[i, :, :] - np.repeat(np.mean(Senales[i, :, :], axis=1).reshape(9000, 1), nelec, axis=1)
     t=np.arange(-lti,len(Senales[0, :, 0])/2000-lti, 1/2000 )
     tindex=[np.argmin(np.abs(t)), np.argmin(np.abs(t-2)), np.argmin(np.abs(t-2.7))]
     if len(elecs)==0:
         elecs=np.zeros((2), dtype=np.int8)
         elecs[0]=0
         elecs[1]=nelec
     for elec in elecs:
        namedir="m%d_s%d_o%d_e%d"%(monkey, serie, orden, elec)
        pathdir=join(filename, namedir)
        if not lexists(pathdir): # El directorio no existe
            chdir(filename)
            mkdir(pathdir) # Crea el directorio
            chdir(originalpath)
        #fig, ax= plt.subplots(1, 2, figsize=(12, 9))
        fig=plt.figure(figsize=(12, 9))
        fig.suptitle(namedir)
        spec=gridspec.GridSpec(nrows=3, ncols=2)
        ax0=fig.add_subplot(spec[0:2, 0])
        vmin=np.min(Senales[:, :, elec-1])
        vmax=np.max(Senales[:, :, elec-1])
        im=ax0.imshow(Senales[PsicSort, :, elec-1], aspect='auto', origin="lower", extent=[t[0], t[-1], 1, len(PsicSort)+1], vmin=vmin, vmax=vmax) 
        ax0.plot((0, 0, 0.5, 0.5), (1, len(PsicSort), len(PsicSort), 1), c="white", ls="--", lw=1)
        # Create divider for existing axes instance
        divider3 = make_axes_locatable(ax0)
        # Append axes to the right of ax3, with 20% width of ax3
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = plt.colorbar(im, cax=cax3)
        cbar3.set_label("Amplitude [$\mu$V]", fontsize=14)
        ax0.set_xlabel("Time [s]", fontsize=14)
        ax0.set_ylabel("# Ordered Trials", fontsize=14)
        ax0.tick_params(labelsize=12)
        ax0.set_title("Original")
      # Ahora viene la componente de selección de ensayos problemáticos  
        extract=np.concatenate((Senales[:, tindex[1]: tindex[2], elec-1].flatten(), Senales[:, 0: tindex[0], elec-1].flatten() ))
        mean=np.median(extract)
        std=(np.quantile(extract, 0.75)- np.quantile(extract, 0.25))/2
        #thr=(np.quantile(extract, 0.995), np.quantile(extract, 0.005))
        thr=(mean + 5*std, mean-5*std)
        ax2=fig.add_subplot(spec[2, :])
        ax2.hist(Senales[:, 0:tindex[0], elec-1].flatten())
        ax2.axvspan(thr[1], thr[0], facecolor="gray", alpha=0.5)
        if not (elec in SpikesLists[:, 1]):
            print("Electrodo %d no se encuentra en las espigas."%(elec))
        else:
            unidades=SpikesLists[np.where(SpikesLists[:, 1]==elec)[0],  2]
            spikeTrials=[]
            for uni_i in unidades:
               try:
                   tmp=np.loadtxt(join(filespikes, "RR0330%d_00%d_e%d_u%d.csv"%(serie, orden, elec, uni_i)), usecols=(0), delimiter=",")
               except OSError:   # File does not exists
                   print("El electrodo ", elec, "y unidad ", uni_i ,"no está")
                   continue
               except:
                   tmp=np.loadtxt(join(filespikes, "RR0330%d_00%d_e%d_u%d.csv"%(serie, orden, elec, uni_i)), usecols=(0), delimiter="\t")
               if len(tmp)<defaultntrials:
                   spikeTrials.append(findabsent(tmp))
            if len(spikeTrials)>0:
                spikeTrials=np.hstack(spikeTrials)
                print("Los ensayos ausentes en las espigas son: ", np.unique(spikeTrials), "electrodo: ", elec)
            else:
                print("No hay ensayos ausentes en las espigas del electrodo ", elec)
        trial=[]
        trial_or=[]
        c=0
        for trial_i in range(ntrials):
            exceso=np.sum(Senales[trial_i, 0:tindex[0], elec-1]>=thr[0]) + np.sum(Senales[trial_i, 0:tindex[0], elec-1]<=thr[1])
            if exceso>800: # 0.4 Segundos es el límite.
                c+=1
                failtrials.append(trial_i)
                print("Problema en el ensayo ",trial_i,". En grafico: ",np.where(PsicSort==trial_i)[0]+1 , " y electrodo ", elec, "Tiempo: ", exceso/2000, exceso)
                trial.append(trial_i)
                trial_or.append(np.where(PsicSort==trial_i)[0])
        if c==0:
            print("Sin problemas en el lfp del electrodo ", elec)
        # Ahora  a revisar los ensayos perdidos en las espigas 
        selectrials=np.setdiff1d(np.arange(ntrials), np.array(trial_or))   # Diferencia de conjuntos
        ax1=fig.add_subplot(spec[0:2, 1])
        im=ax1.imshow(Senales[PsicSort[selectrials], :, elec-1], aspect='auto', origin="lower", extent=[t[0], t[-1], 1, len(selectrials)+1], vmin=vmin, vmax=vmax) 
        ax1.plot((0, 0, 0.5, 0.5), (1, len(selectrials), len(selectrials), 1), c="white", ls="--", lw=1)
        ax1.set_title("Limpio")
        # Create divider for existing axes instance
        divider3 = make_axes_locatable(ax1)
        # Append axes to the right of ax3, with 20% width of ax3
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = plt.colorbar(im, cax=cax3)
        cbar3.set_label("Amplitude [$\mu$V]", fontsize=14)
        ax1.set_xlabel("Time [s]", fontsize=14)
        ax1.tick_params(labelsize=12)
        fig.tight_layout() 
     return Senales, PsicSort, Psic, selectrials, failtrials

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
##                                  Bloque sobre el ICA
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

@njit()
def g(x):
    return np.tanh(x)

@njit()

def g_der(x):
    return 1 - g(x) * g(x)


def center(x):
    x = np.array(x)
    
    mean = x.mean(axis=1, keepdims=True)
    
    return x - mean

def whitening(x, return_inv=False):
    """
    Funcion que permite realizar parte del preprocesamiento de la matriz
    de senales.
    """
    cov = np.cov(x)

    d, E = np.linalg.eigh(cov)

    D = np.diag(d)

    D_inv = np.sqrt(np.linalg.inv(D))

    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))
    if not return_inv:
        return x_whiten
    else:
        return x_whiten, D, E


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w

    w_new /= np.sqrt((w_new ** 2).sum())

    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    
    X = whitening(X)
        
    components_nr = X.shape[0]

    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):
        
        w = np.random.rand(components_nr)
        
        for j in range(iterations):
            
            w_new = calculate_new_w(w, X)
            
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            
            w = w_new
            
            if distance < tolerance:
                break
                
        W[i, :] = w
        
    S = np.dot(W, X)
    
    return S
                            
def escala(senales):
    """
    Esta funciÃ³n escala la matriz de senales, es decir les divide la desviaciÃ³n estÃ¡ndar.

    Parameters
    ----------
    senales : 2nd array. 
        Se trata de la matriz de seÃ±ales, cada renglÃ³n es una seÃ±al
        
    Returns
    -------
    None.

    """
    stdev=np.std(senales, axis=1)
    return senales/stdev

def loadLFPraw(area, monkey, serie, orden, lti=1.5, duracion=4.5, computador=1, **kwargs):
    """
    Esta función nos devuelve las señales de LFP recortadas, alineadas y filtradas con un
    filtro pasabajas con frecuencia de corte en 120. Esta función no filtra en 60 Hz. 
    
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
    ntrials=len(datalfp)                # Número de ensayos en la sesión 
    #------------------------------------------------------------------------------
    if area=="S1":
        try:
            Psic=np.loadtxt("%s/Text_s/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter=",")            
            #Psic=np.loadtxt("/home/sparra/AENHA/Database_RR033/Text_s/RR033075_001/RR033075_001_Psyc.csv",delimiter="\t")
        except:
            Psic=np.loadtxt("%s/Text_s/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter="\t")
            #Psic=np.loadtxt("/home/sparra/AENHA/Database_RR033/Text_s/RR033075_001/RR033075_001_Psyc.csv",delimiter=",") 
        #Psic=np.loadtxt("/run/media/sparra/AENHA/Database_RR033/Text_s/RR033091_002/RR033091_002_Psyc.csv",delimiter="\t")
        try:
            Tiempos=np.loadtxt("%s/Text_s/%s/%s_T.csv"%(pathcomp, name, name), delimiter="\t")
            #Tiempos=np.loadtxt("/home/sparra/AENHA/Database_RR033/Text_s/RR033075_001/RR033075_001_T.csv",delimiter="\t")
        except:
            Tiempos=np.loadtxt("%s/Text_s/%s/%s_T.csv"%(pathcomp, name, name), delimiter=",")
            #Tiempos=np.loadtxt("/home/sparra/AENHA/Database_RR033/Text_s/RR033075_001/RR033075_001_T.csv",delimiter=",")
    else:
        Psic=np.loadtxt("%s/Text_sA1/%s/%s_Psyc.csv"%(pathcomp, name, name), delimiter=",")
        Tiempos=np.loadtxt("%s/Text_sA1/%s/%s_T.csv"%(pathcomp, name, name), delimiter=",")
    #Tiempos=np.loadtxt("/run/media/sparra/AENHA/Database_RR033/Text_s/RR033077_001/RR033077_001_T.csv",delimiter="\t")
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    ## Cabeceras y constantes importantes para la función
    T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
    P={"Ntrial":0, "Nclase":1, "Answer":2, "CAnswer":3, "Hit?":4, "ATac1":5, "frec1":6, "AAud1":7, "ATac2":8, "frec2":9, "AAud2":10, "ATac3":11, "frec3":12, "AAud3":13, "RxKd":14, "RxKU":15, "RxPB":16, "BT":17, "TRW":18, "Set":19, "Ronda":22}

    #print("Los intervalos umbrales van por encima del 25% y menor al 75%")
    #print("Los intervalos subumbrales van por debajo del 25% ")

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    ## Empecemos por analizar los lfp de los ensayos subumbrales

    # Parámetro del filtro pasabajas

    Senales=np.zeros((ntrials, longitud, nelec))
    for trial in range(len(datalfp)):
        inicio=np.int32((Tiempos[trial, T["iE3"]]-lti)*fsamp)
        for elec in range(nelec):    
            # Filtado pasabajas
            Senales[trial, :, elec]=datalfp[trial][inicio:inicio+longitud, elec]         

    return Senales

def plotcomp(grupo1, grupo2, t=( ), title=("Grupo1", "Grupo2"), **kwargs):
    """
    
    Esta figura realiza un plot con las se\~nales de dos grupos en un orden invertido (de arriba hacia abajo).
    Las se\~nales graficadas ser\'an previamente llevadas a tener media cero y varianza 1.
    
    Parameters
    ----------
    grupo1 : 2d-numpy array. Es una matrix donde los
    renglones representan distintas seÃ±ales.
    
    grupo2 : 2d-numpy array. Es una matrix donde los
    renglones representan distintas seÃ±ales.

    Returns
    -------
    None.

    """
    fig, ax=plt.subplots(1, 2, figsize=(15, 9))
    nsenales, lent=grupo1.shape
    nsenales2, lent2=grupo2.shape
    std=(grupo1.std(), grupo2.std())
    if nsenales!=nsenales2 or lent!=lent2:
        raise ValueError("Las dimensiones de los grupos 1 y 2 deben coincidir")
    if len(t)==0:
        t=np.linspace(-1.5, 3, lent)
    elif len(t)!=lent:
        raise ValueError("Las dimensiones del arreglo t debe coincidir con las columnas de grupo1 y grupo2")
    basal=np.array([0, 0], dtype=np.float32)
    for i in range(grupo1.shape[0]):
        ax[0].plot(t, grupo1[i, :] + basal[0], c="black")
        ax[1].plot(t, grupo2[i, :] + basal[1], c="black")
        basal[0]+=np.max(np.abs(grupo1[i, :]))+ std[0]*2
        basal[1]+=np.max(np.abs(grupo2[i, :]))+ std[1]*2
    count=0
    for ax_i in ax.flatten():
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_i.spines["left"].set_visible(False)
        ax_i.tick_params(labelsize=16)
        ax_i.set_xlabel("Tiempo [s]", fontsize=16)
        ax_i.axvspan(0, 0.5, facecolor="gray", alpha=0.5)
        ax_i.set_title(title[count])
        count+=1
    fig.tight_layout()
    plt.show()
    return None


#- - - - - - - - - - - - - - Aquí es donde se meterían las llamadas al gui- - - - - - - - - - - - - - - - - - - - - - 

Params=InputParameters(monkey, serie,  orden, electrodes, computador, lti)     # Caracteristicas de la seria a analizar   

Senales, PsicSort, Psic, selectrials, failtrials=plotoriginals(Params.m, Params.s, Params.o,
              area, Params.e, Params.c, lti=Params.lti)
Senales_raw= loadLFPraw(area, Params.m, Params.s, Params.o, lti=1.5, duracion=4.5, computador=Params.c) 
# Selección de ensayos problemáticos en 4 electrodos o más
failtrials, Nfailtrials=np.unique(failtrials, return_counts=True)
tmpindices=np.where(Nfailtrials>=3)[0]
if len(tmpindices)==0:
    failtrials=[]
else:
    failtrialsdep=failtrials[tmpindices]
    Nfailtrials=Nfailtrials[tmpindices]
del tmpindices

goodtrials=tuple(i for i in range(150) if not( i in(failtrials)) )    
t=np.linspace(-Params.lti, Params.lti+3, 9000)
# Calcula la transformada de Fourier y muestra las gráficas de todos los electrodos, en cada figura todos los ensayos.
SenalesX=Fourier(Senales[PsicSort], fsamp=2000)
freqs=np.linspace(0, 1000, len(t)//2 + 1)
freqsshort=np.where(freqs<=80)[0]
ntrials, lent, nelec=Senales.shape
for elec_i in range(nelec):
    fig_i, ax_i=plt.subplots(1,1, figsize=(9, 6))
    im=ax_i.imshow(SenalesX[:, :, elec_i], aspect="auto", origin="lower", extent=[0, 1000, 1, ntrials +1], cmap="seismic",
                   vmin=np.quantile(SenalesX[:, :, elec_i].flatten(), 0.15) , vmax=np.quantile(SenalesX[:, :, elec_i].flatten(), 0.95))
    divider3 = make_axes_locatable(ax_i)
    # Append axes to the right of ax3, with 20% width of ax3
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im, cax=cax3)
    ax_i.set_xlim([0, 80])
    ax_i.set_xlabel("Frequencies [Hz]", fontsize=10)
    ax_i.set_ylabel("# Ordered Trials", fontsize=10)
    ax_i.set_title("Electrodo " + str(elec_i + 1), fontsize=12)


#Muestra de los ensayos problemáticos
ica = FastICA(n_components=7, random_state=0, tol=0.01)
Senales_Limpias=np.copy(Senales)
for index in failtrials[12::]:
    # print("=========================================================",
    #       "Selecciona el ensayo con el cual se iniciará la limpieza:",
    #       "=========================================================")
    # print(failtrials)
    # index=np.int16(input(": "))
    # #for index in failtrials:
    #senal_white=np.copy(Senales_raw[index, :, :])
    
    while(True):
        senal_white=np.copy(Senales[index, :, :])
        for elec_i in range(nelec):
            senal_white[: , elec_i]=senal_white[: , elec_i]- np.mean(senal_white[: , elec_i])
            #senal_white[: , elec_i]=senal_white[: , elec_i]/np.std(senal_white[: , elec_i])
        senal_cov0, D, E=whitening(senal_white.transpose(), return_inv=True)               # Covarianza es cero
        comps = ica.fit_transform(senal_cov0.transpose())           #Descomposición en componentes independientes
        plotcomp(comps.transpose(), Senales_raw[index, :, :].transpose(), t, title=("ICA", "Gsenal"))
        print("¿Cuáles son los componentes que se desvanecerá? -1 para salir o si se repite")
        whica_i=0
        whica=[]
        while(whica_i!=-1  ):
            whica_i=np.int8(input(":")) 
            if not (whica_i in whica) and whica_i!=-1:
                whica.append(whica_i)
            else: 
                break
        del whica_i
           
        comps[:, whica]=0
        restored = ica.inverse_transform(comps)
        plotcomp(senal_cov0, restored.transpose(), t, title=("Señal blanqueada", "Señal restaurada"))
        #plotcomp(Senales[index, :, :].transpose(), restored.transpose(), t, title=("Original", "Restaurada"))
        senal_cov0_unw=np.dot(E, np.dot(D, np.dot(E.T, senal_cov0) ))
        #plotcomp(Senales[index, :, :].transpose(), senal_cov0_unw, t, title=("Original", "wh-unwh"))
        restored_unw=np.dot(E, np.dot(D, np.dot(E.T, restored.transpose()) ))
        Senales_Limpias[index, :, :]=np.copy(restored_unw.transpose())
        #plotcomp(senal_cov0, restored, t, title=("Original-Wh", "Restaurada_wh"))
        plotcomp(Senales[index, :, :].transpose(), restored_unw, t, title=("Original", "Limpia-final"))
        restored_unwX=Fourier(restored_unw.reshape(1, 9000, 7), showfig=False)
        plotcomp(SenalesX[index, freqsshort, :].transpose(), restored_unwX[0, freqsshort, :].transpose(), freqs[freqsshort], title=("Original", "Limpia-final"))
        if bool(np.int8(input("Satisfecho? 0 No, otro sí: "))):
            break

            
# Ahora a comparar los registros:    




# Segunda aproximación mediante el uso de la descomposición empírica de modos
# imf=[]
# for elec_i in range(nelec):
#     tmp=emd(Senales[index, :, elec_i] , maxorder=10, maxstd=0.5, maxite=5000)
#     imf.append(tmp)

# plotcomp(imf[0], imf[1], t, title=("IMF 1 elc", "IMF 2 elc"))


# Ahora viene el cálculo de la  ica para la limpieza  
# for i in range(ns):
#    #savename=join(pathdir, "RR033%d_00%d"%(Parametros[i].s, Parametros[i].o))
#     Data=[]
#     for elec_i in Parametros[i].e:
#        Data.append(loadlfplimpio(area, Parametros[i].m, Parametros[i].s, Parametros[i].o, elec_i, computador=computador))
# # PreparaciÃ³n de los grupos de datos para la comparaciÃ³n visual previa a la limpieza
#     lent=Data[0].shape[1]-23   # Son 23 los elementos en la psicofÃ­sica
#     nelec=len(Parametros[i].e)
#     ntra=len(Parametros[i].tra)
#     ntrb=len(Parametros[i].trb)
#     grupo1=np.zeros((nelec*ntra, lent), dtype=np.float32)
#     grupo2=np.zeros((nelec*ntrb, lent), dtype=np.float32)
#     count=0
#     for tr_i in range(ntra):
#         for elec_i in range(nelec):          
#             grupo1[count, :]=Data[elec_i][Parametros[i].tra[tr_i], 23::]- np.mean(Data[elec_i][Parametros[i].tra[tr_i], 23::])
#             grupo1[count, :]=grupo1[count, :]/np.std(grupo1[count, :])
#             count+=1
#     count=0
#     for tr_i in range(ntrb):
#         for elec_i in range(nelec):
#             grupo2[count, :]=Data[elec_i][Parametros[i].trb[tr_i], 23::] - np.mean(Data[elec_i][Parametros[i].trb[tr_i], 23::])
#             grupo2[count, :]=grupo2[count, :]/np.std(grupo2[count, :])
#             count+=1
#     t=np.linspace(-1.5, 3, 9000)
#     plotcomp(grupo1, grupo2, t)


# Ahora viene la segunda parte respecto a la obtenci\'on de diversas fuentes



# ica = FastICA(n_components=7, random_state=0, tol=0.01)
# pca=PCA(n_components=7 )
# Descomp=[]
# for i in range(4):
#     begin=i*7  
#     tmp=whitening(grupo1[begin:begin+7, :])
#     tmp=tmp.transpose()
#     comps = ica.fit_transform(tmp)
#     pcadata=pca.fit_transform(tmp)
#     plotcomp(comps.transpose(), pcadata.transpose(), t)
#     comps2=ica.fit_transform(pcadata)
#     plotcomp(comps2.transpose(), comps.transpose(), t)
#     Descomp.append(comps.transpose())
    
#     restored = ica.inverse_transform(comps)

#     restored2 = ica.inverse_transform(comps2)
# Descomp=np.vstack(restored)
# plotcomp(tmp.transpose(), restored2.transpose(), t)

# plotcomp(grupo1[begin:begin+7, :], restored.transpose(), t)


