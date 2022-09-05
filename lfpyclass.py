# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:48:31 2022

@author: sparra
"""
from os import name as nameos
import numpy as np
import Phd_extV2 as Pext
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


def sigmoidal(X, a, b, c, d):
    return (float(a-b))/(1.0 + np.exp((float(d)- X.astype(float))/float(c))) + float(b)

def fitmatlab(x):
    from numpy import log
    return 9.904*log(96.58*x+12.75) + 4.749

# class MyError is extended from super class Exception
class Not_def_func(Exception): #User_Error
   # Constructor method
   def __init__(self, value):
      self.value = value
   # __str__ display function
   def __str__(self):
      return(repr(self.value))
# try:
#    raise(Not_def_func("Función no definida"))
#    # Value of Exception is stored in error
# except Not_def_func as error:
#    print('Ocurrió una nueva excepción :',error.value)

class lfps_ps:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None
    
    def loadspikes(self, electrode, unit):
        if nameos=="posix": # Puede ser el computador hp o lenovo
            if self.serie>100:   ## Mono 32
                path="/home/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s/RR032%d_002/RR032%d_002"%(self.serie, self.serie)
            else:             ## Mono 32
                path="/home/sparra/AENHA/Database_RR033/Text_s/RR0330%d_00%d/RR0330%d_00%d"%(self.serie, self.orden, self.serie, self.orden)
        else: # Computador del lab            
            if self.serie>100:   ## Mono 32
                path="D:\\BaseDatosKarlitosNatsushiRR032\\Text_s\\RR032%d_002\\RR032%d_002"%(self.serie, self.serie)
            else:             ## Mono 32
                path="D:\\Database_RR033\\Text_s\\RR0330%d_00%d\\RR0330%d_00%d"%(self.serie, self.orden, self.serie, self.orden)
    
        try:
            A=np.loadtxt(path+ "_e%d_u%d.csv"%(electrode, unit), delimiter=",", usecols=(0))
            delimiter=","
        except IOError:
            print("El archivo %s no existe"%(path + "_e%d_u%d.csv"%(electrode, unit)))
            return -1
        except:
            A=np.loadtxt(path+ "_e%d_u%d.csv"%(electrode, unit), delimiter="\t", usecols=(0))
            delimiter="\t"
        spikes=Pext.SLoad(path+ "_e%d_u%d.csv"%(electrode, unit), delimiter)
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
        # Agregamos atributos

        # self.times=times
        # self.psyc=psyc
        # self.spks=spikes
        return times, psyc, spikes  
       
    def dict2ndarray(self):
        """
        This method get the signals per trial and electrode. from the dictionary 
        format in a 3d-numpy array. The new array uses the rows as time and columns
        as electrodes, third dimmension are the trials. If any of the electrodes 
        have not signal recorded is filled with NANs

        Returns
        -------
        3d-numpy array.

        """
        r,c,d=0,0,0
        for keys, values in  self.senales.items():
            d+=1
            rtmp,ctmp=values.shape  
            if rtmp>r:
                r=rtmp
            if ctmp>c:
                c=ctmp
        senales3d=np.zeros((r, c, d), dtype=np.float32)
        for idx, keys_i in enumerate(self.senales.keys()):
            senales3d[:, :, idx]=np.copy(self.senales[keys_i])
        return senales3d
       
    def loadspks(self):
        """
        Loads spikes related to session and monkey in the object created.
        Returns
        -------
        None.

        """
        spikes={}
        flag=True
        for idx, elec_i in enumerate(self.electrodes):
            spk=[]
            for unit_i in (self.neurons[idx]):
                if flag:
                    times, psyc, tmp=self.loadspikes(elec_i, unit_i)
                    spk.append(tmp)
                    flag=False
                else:
                    _, _, tmp=self.loadspikes(elec_i, unit_i)        
                    spk.append(tmp)
            spikes[str(elec_i)]=spk.copy()
        setattr(self, "times",  times)
        setattr(self, "psyc",   psyc)
        setattr(self, "spikes", spikes)
        return None
    
    
    def loadsignal(self,  computador, fun=(1,), **kwargs ):
        senales={}
        if type(fun)==tuple: # No es una función, usamos las prehechas
            if fun[0]==1: # Usamos loadLFP
                self.loadLfp(computador=computador, **kwargs)            
            else:         # Usamos loadlfplimpio
                for electrode_i in self.electrodes:
                    tmp=self.loadlfplimpio(electrode_i, computador=computador)
                    senales[str(electrode_i)]=tmp.copy()
        else:
            raise Not_def_func("Componente aún no soportada upss")
        
        setattr(self, "senales", senales)
        time_sgnl=np.linspace(-self.lti, self.dur-self.lti, np.int32(self.dur*self.fsamp))
        setattr(self, "time_sgnl", time_sgnl)   # Set time attr for signals.
        
        
    def loadLFP(self,  lti=1.5, duracion=4.5,  return_psych=False, computador=1,  **kwargs): # monkey, serie, orden, electrodes, neurons
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
        #fsamp=2000                          # La frecuencia de muestreo es de 2khz
        longitud=np.int32(duracion*self.fsamp)   # Tiempo en núm de puntos
        nelec=7                             # Número de electrodos
        #widthtf=0.5                         # Ancho de las ventanas en el espectrograma
        
        if self.area=="S1":
            patharea="S1_izq_MAT"
        elif self.area=="A1":    
            patharea="A1_izq_MAT"
        else:
            raise ValueError("Hasta el momento solo se está trabajando con las áreas S1 y A1")
    
        if self.monkey==33:
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
            if self.serie<100:
                name="RR0330%d_00%d"%(self.serie, self.orden)
            elif self.serie<10:
                name="RR03300%d_00%d"%(self.serie, self.orden)
            else:
                name="RR033%d_00%d"%(self.serie, self.orden)
        elif self.monkey==32:
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
            if self.serie<100:
                name="RR0320%d_00%d"%(self.serie, self.orden)
            elif self.serie<10:
                name="RR03200%d_00%d"%(self.serie, self.orden)
            else:
                name="RR032%d_00%d"%(self.serie, self.orden)
    
        else:
            raise ValueError("Valor incorrecto, solo es posible trabajar con los datos del mono 32 y 33")
        #------------------------------------------------------------------------------
    
        # Carga un archivo de prueba
        file="%s/%s"%(path, name)
        #file="/run/media/sparra/AENHA/LFP33/S1_izq_MAT/RR033075_001"
        if (self.monkey==32 or self.monkey==33) and self.area=="S1":
            datalfp=lfp.loadlfp(file)
            if len(datalfp)==0:
                print("Load file with loadlfp method failed .... \ntrying with loadnsx method...")
                datalfp, _=lfp.loadnsx(file)
        elif (self.monkey==32 or self.monkey==33) and self.area=="A1":
            datalfp, _=lfp.loadnsx(file)
        else:
            raise ValueError("Monkey or area not supported.")
        #------------------------------------------------------------------------------
        ntrials=len(datalfp)                # Número de ensayos en la sesión 
        #------------------------------------------------------------------------------
    
        if self.area=="S1":
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
        orden=np.int32(tam/fcut*self.fsamp)            # Orden del filtro pasabajas
        Q=90                                      # Factor de calidad del filtro notch (Remueve el ruido de linea [60 hz])
        b=sn.firwin(orden, 2*fcut/self.fsamp)              # Coeficientes para el filtro pasa bajas
        Senales=np.zeros((ntrials, longitud))
        a_notch, b_notch,=sn.iirnotch(2*60/self.fsamp, Q)
        for trial in range(len(datalfp)):
            inicio=np.int32((Tiempos[trial, T["iE3"]]-lti)*self.fsamp)
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
        setattr(self,"senales", Senales)
        setattr(self,"thresholding",  (maskSupra_A, maskSupra_T, maskUmb_A, maskUmb_T, maskSub_A, maskSub_T ,maskzero))
        return None
    
    def loadlfplimpio(self, electrode, computador=1,  **kwargs):
        """
        This fuction loads data and make the cut according to the -lti attribute where 0 is at the 
        stimulus onset and dur is the attribute that indicates the length in
        sec. of the signals. Data loaded are fixed in this values. If another values are needed
        then it is necessary to use the other load data functions
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
        Data. Returns the signals aligned to the stimulus onset and with the 
              duration marked by dur.
        """
        self.dur=4.5    # Set the attrbute in the value corresponding to
                        # data loaded in this method
        if self.area=="S1":
            patharea="S1_Izq_txt"
        elif self.area=="A1":    
            patharea="A1_Izq_txt"
        else:
            raise ValueError("Hasta el momento solo se está trabajando con las áreas S1 y A1")
        if self.monkey==33:
            if computador==1: #Lap Lenovo
                path="/home/sparra/AENHA/LFP33/%s/"%(patharea)
            elif computador==2: # Lap Hp
                path="/run/media/sparra/AENHA/LFP33/%s/"%(patharea)
            else:  #Lab
                path="D:\\LFP33\\%s\\"%(patharea)
            #namedir="RR033"
            name="m33_s%d_o%d_e%d.csv"%(self.serie, self.orden, electrode)
    
        elif self.monkey==32:
            if computador==1: #Lnv
                path="/home/sparra/AENHA/LFP32/%s/"%(patharea)
            elif computador==2:  #Hp
                path="/run/media/sparra/AENHA/LFP32/%s/"%(patharea)
            else:    #Lab
                path="D:\\LFP32\\%s\\"%(patharea)
            name="m32_s%d_o%d_e%d.csv"%(self.serie, self.orden, electrode)
        else:
            raise ValueError("Valor incorrecto, solo es posible trabajar con los datos del mono 32 y 33")
        Data=np.loadtxt(path + name, delimiter=",", skiprows=(1)) # Load file
        
        ntrials, _=np.shape(Data)  # Trials number and length signal
        if not hasattr(self, "psyc"):
            self.loadspks()
        ## Cabeceras y constantes importantes para el script
        T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
    
        #------------------------------------------------------------------------------
        # masktac=Psic[:, P["AAud3"]]>0    
        # Psic[masktac, P["AAud3"]]=fitmatlab(Psic[masktac, P["AAud3"]])
        # #Ahora viene el ordenamiento de los ensayos táctiles. Amplitud de menor a mayor
        # maskSupra_T=np.where(Psic[:, P["ATac3"]]>=Thresholding[0, 1])[0]
        # #Ahora viene el ordenamiento de los ensayos acústicos. Amplitud de menor a mayor
        # maskSupra_A=np.where(Psic[:, P["AAud3"]]>=Thresholding[1, 1])[0]
        # maskzero=np.where((Psic[:, P["ATac3"]]==0)*(Psic[:, P["AAud3"]]==0))[0]
        # #Ahora viene el ordenamiento de los ensayos táctiles. Subumbrales
        # maskSub_T=np.where((Psic[:, P["ATac3"]]<=Thresholding[0, 0])*(Psic[:, P["ATac3"]]>0)  )[0]
        # #Ahora viene el ordenamiento de los ensayos acústicos. Subumbrales
        # maskSub_A=np.where( (Psic[:, P["AAud3"]]<=Thresholding[1, 0])*(Psic[:, P["AAud3"]]>0) )[0]
        # #Ahora viene el ordenamiento de los ensayos táctiles. Umbrales
        # maskUmb_T=np.where( (Psic[:, P["ATac3"]]>Thresholding[0, 0])*(Psic[:, P["ATac3"]]<Thresholding[0, 1])  )[0]
        # #Ahora viene el ordenamiento de los ensayos acústicos. Umbrales
        # maskUmb_A=np.where((Psic[:, P["AAud3"]]>Thresholding[1, 0])*(Psic[:, P["AAud3"]]<Thresholding[1, 1]))[0]    
        # setattr(self,"senales", Senales)
        # setattr(self,"thresholding",  (maskSupra_A, maskSupra_T, maskUmb_A, maskUmb_T, maskSub_A, maskSub_T ,maskzero))
         
        return Data
    
    def get_thresholds(self):
        """
        THis method calculates thresholds for indicating the intervals corresponding
        for sub-threshold, threshold, and supra-threshold values in the 
        acoustic and vibrotactil modality 
        
        """
        if not hasattr(self, "thr"):
            def sigmoidal_ac(X,a,b,c,d):
                return a-(1-a-b)*(1/np.pi * np.arctan((X-c)/d) + 0.5 )
            Tac_psic=(0.796406, 0.037770, 1.495931, 10.667152)
            #Ac_psic=(0.891063, 0, 1.952362, 34.102654)
            Ac_psic=( 0.106715, 1.754315, 40.000000, 2.000000)
            xtac=np.linspace(0, 28, 500)
            xac=np.linspace(0, 69, 500)
        
            Thresholding=np.zeros((2, 3), dtype=np.float32)
            Thresholding[0, 1]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.75))]
        
            Thresholding[1, 1]=xac[np.argmin(np.abs(sigmoidal_ac(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.75))]
            # Ahora para el umbral 0.5
            Thresholding[0, 2]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.5))]
        
            Thresholding[1, 2]=xac[np.argmin(np.abs(sigmoidal_ac(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.5))]
        
            #print("Los intervalos supraumbrales van por encima del 75%")
            Thresholding[0, 0]=xtac[np.argmin(np.abs(sigmoidal(xtac, Tac_psic[0], Tac_psic[1], Tac_psic[2], Tac_psic[3])-0.25))]
        
            Thresholding[1, 0]=xac[np.argmin(np.abs(sigmoidal_ac(xac, Ac_psic[0], Ac_psic[1], Ac_psic[2], Ac_psic[3])-0.25))]
            #print("Los intervalos umbrales van por encima del 25% y menor al 75%")
            #print("Los intervalos subumbrales van por debajo del 25% ")
            setattr(self, "thr", {"tac": Thresholding[0, :],
                                        "acous":Thresholding[1, :]})
        return None
    
    