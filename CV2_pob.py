#Carga librerías
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap=cm.get_cmap("tab20")
import Phd 
import Phd_ext as Pex
from scipy.signal import savgol_filter
#-______-Parámetros del script
wnd=0.3
stp=0.010

tini=-1.- wnd
tfin=2 + wnd

CR=3   #Indica el tipo de campo receptor
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#----------------------------------          Define constantes para el manejo de archivos   ----------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
#Diccionario indicador de las cabeceras
headers={"Mono":0, "Serie":1, "Profundida":2, "Elec":3, "Uni":4, "setN":5, "SetId":6, "CR":7, "Elec_cuidado":8, "Adaptador":9, "3b":10, "Hits":11, "Amp":12, "Trial":13, "Data":14}

#por neurona (-1+0.05)/0.010   (shift+wnd)/step
#--------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------      Lectura de archivos  ------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
root="/home/sparra/MEGA/Prjl"
#root="/home/sparra/Prjl"
R33=np.loadtxt(root+"/RR0330Tasa_individual.csv", delimiter=',', usecols=( (i for i in range (13))))
RR032_a=np.loadtxt(root+"/RR032Tasa_individual_RR032P1.csv", delimiter=',', usecols=((i for i in range (13))))
RR032_b=np.loadtxt(root+"/RR032Tasa_individual_RR032P2.csv", delimiter=',', usecols=((i for i in range (13))))
RR032_c=np.loadtxt(root+"/RR032Tasa_individual_RR032P3.csv", delimiter=',', usecols=((i for i in range (13))))
R32=np.concatenate((RR032_a, RR032_b, RR032_c), axis=0)
R32=np.concatenate( (  np.ones( (np.size(R32, 0), 1))*32, R32), axis=1)
R33=np.concatenate( (np.ones((np.size(R33, 0), 1))*33, R33), axis=1)
neuronas32=np.unique(R32[:, 0: 11], axis=0)
neuronas33=np.unique(R33[:, 0:11], axis=0)
neuronas=np.concatenate((neuronas32, neuronas33), axis=0)
Nneu=np.size(neuronas, axis=0)
neuronas=neuronas.astype(np.int16)
Amplitudes=np.unique(np.concatenate( (R32[:, headers["Amp"]], R33[:, headers["Amp"]])   ))

del RR032_a, RR032_b, RR032_c, R32, R33, neuronas32, neuronas33  #Destruct objects
nelem=int(np.ceil((tfin-tini)/stp)-wnd/stp+1)
N_wnd_Pob_dict={}
CV2_wnd_Pob_dict={}
#---------------- Empieza la lectura de archivos y el cálculo de el cálculo de variación.------------------
for b3 in range(2):   #Indica el tipo de neurona, 0 A1, 1 3b
    print("Inicia el análisis de las neuronas de CR=%d, y Area %d"%(CR, b3))    
    posCR1=np.where( (neuronas[:, headers["CR"]]==CR)*(neuronas[:, headers["3b"]]==b3) )[0]
    Nneu=len(posCR1)  # Para CR1 ambas poblaciones, maximo 125
    N_wnd_Pob=np.zeros((len(Amplitudes) , nelem))
    CV2_wnd_Pob=np.zeros((len(Amplitudes) , nelem))
   
    for j in range(Nneu):
        pos=posCR1[j]
        if neuronas[pos, 1]>100:
            filename="RR0" + str(neuronas[pos, 0] ) + str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])+"_e"+str(neuronas[pos, 3]) + "_u"+str(neuronas[pos, 4])+".csv"
            fname="RR0" + str(neuronas[pos, 0] ) + str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])
        elif neuronas[pos, 1]>10:
            filename="RR0" +  str(neuronas[pos, 0] ) + "0" + str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])+"_e"+str(neuronas[pos, 3]) + "_u"+str(neuronas[pos, 4])+".csv"
            fname="RR0" + str(neuronas[pos, 0] ) + "0" +  str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])
        elif neuronas[pos, 1]<10:
            filename="RR0" + str(neuronas[pos, 0] ) + "00" +  str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])+"_e"+str(neuronas[pos, 3]) + "_u"+str(neuronas[pos, 4])+".csv"
            fname="RR0" + str(neuronas[pos, 0] ) + "00" + str(neuronas[pos, 1]) + "_00" + str(neuronas[pos, 5])
        if neuronas[pos, 0]==32:
            filesdir="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s/"+fname
        else:
            filesdir="/run/media/sparra/AENHA/Database_RR033/Text_s/"+ fname
        try:    
            sep=','
            ensayos=np.loadtxt(filesdir +"/" +  filename, usecols=(0), delimiter=sep)
        except:
                sep="\t"
                ensayos=np.loadtxt(filesdir +"/" +  filename, usecols=(0), delimiter=sep)
        ensayos=ensayos.astype(np.int32)-1        
        Spikes=Pex.SLoad(filesdir + "/" +  filename, sep)

        filename=fname +"_Psyc.csv"
        try:
            Psicof=np.loadtxt(filesdir + "/"+ filename, delimiter="\t")
        except:
            Psicof=np.loadtxt(filesdir + "/"+ filename, delimiter=",")
        Psicof=Psicof[ensayos, :]
        filename= fname +"_T.csv"
        try:
            Tiempos=np.loadtxt(filesdir + "/"+ filename, delimiter="\t")
        except:
            Tiempos=np.loadtxt(filesdir + "/"+ filename, delimiter=",")
        Tiempos=Tiempos[ensayos, :]
        #Reseteo de los tiempos al inicio del estímulo principal.
        for i in range(len(Spikes)):
            Spikes[i][1::]=Spikes[i][1::] - Tiempos[i, 6]
        for i in range(len(Amplitudes)):
            if Amplitudes[i]<3:
                amppos=np.where( (Psicof[:, [11]]==0)* (Psicof[:, [13]]==Amplitudes[i] )) [0]
            else:
                amppos=np.where( (Psicof[:, [11]]==Amplitudes[i])* (Psicof[:, [13]]==0))[0]        
            Espigas=np.array(Spikes, dtype=object)  #numpy array de numpy arrays
            Espigassegm=Espigas[amppos]
            Ntr, CV2_zeros=Phd.CV2(Espigassegm, wnd,  stp, tini, tfin)    
            tmp=np.nansum(CV2_zeros, axis=0) #/np.nansum(Ntr, axis=0 )
            tmp[np.isnan(tmp)]=0
            CV2_wnd_Pob[i , :]+=tmp
            tmp=np.nansum(Ntr, axis=0 )
            tmp[np.isnan(tmp)]=0
            N_wnd_Pob[i, :] +=tmp
    key="Ntr-CR-" +str(CR) + "_3b:" + str(b3)
    N_wnd_Pob_dict.update({key:N_wnd_Pob})
    key="CV2-CR-" +str(CR) + "_3b:" + str(b3)
    CV2_wnd_Pob_dict.update({key:CV2_wnd_Pob})


##
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     GRAFICACION    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     GRAFICACION    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     GRAFICACION    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     GRAFICACION    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
##
print("Guardado de archivos ...")
filename="CV2_Pob_CR_%d_wnd_%.3f_stp_%.3f.pkl"%(CR, wnd, stp)
files=open(filename, "wb")
Datos={"Amplitudes_orden":Amplitudes, "wnd":wnd, "stp": stp, "version":"Python 3.8.6"}
Datos.update(N_wnd_Pob_dict)
Datos.update(CV2_wnd_Pob_dict)
pkl.dump(Datos, files)
files.close()
print("Comienza la etapa de graficación...")
colores=["red", "blue", "green", "magenta",   "orange",  "brown", "olive",   "purple",   "pink",  "cyan",  "navy",   "teal", "gray", "black", "yellow", "crimson"]
colorTac=["gold", "darkorange", "orangered", "chocolate", "tomato", "red", "maroon", "firebrick" ]
colorAud=["crimson", "magenta", "purple",  "blueviolet", "teal", "slateblue", "blue", "navy", "slategrey"]
tmp=(Amplitudes<3)*(Amplitudes>0)
Amplitudes[tmp]=Phd.fitmatlab(Amplitudes[tmp])
t=np.arange(tini+wnd, nelem*stp +tini+wnd, stp) 
fig, axes=plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
count=0
#ccolor=1
ccolorTac=0
ccolorAud=0
# Creating keys for plotting
keyCV_A1="CV2-CR-%d_3b:0"%(CR)
keyCV_3b="CV2-CR-%d_3b:1"%(CR)
keyCVN_A1="Ntr-CR-%d_3b:0"%(CR)
keyCVN_3b="Ntr-CR-%d_3b:1"%(CR)
for i in Amplitudes:
    print("Vamos en i: %.2f\t y count=%d  "%(i, count))
    if i>24 or i==0:  # Caso auditivo
        None
    if i in [0, 6, 8, 10, 12, 24]:    #Caso vibrotáctil
        axes[0].plot(t,  savgol_filter( (Datos[keyCV_A1][count, :] )/(Datos[keyCVN_A1][count, :]), 7, 3)   , c=colorTac[ccolorTac]) 
        axes[1].plot(t,   savgol_filter( (Datos[keyCV_3b][count, :] )/(Datos[keyCVN_3b][count, :]), 7, 3)  , c=colorTac[ccolorTac], label="%d $\mu$m"%(Amplitudes[count])) 
        count+=1
        ccolorTac+=1
    else:
        count+=1
plt.show(block=False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_xlabel("Time [s]", fontsize=14)
axes[0].set_title("Neuronas del área 1", fontsize=14)
axes[1].set_title("Neuronas del área 3b", fontsize=14)
axes[0].axvspan(0, 0.5, facecolor="gray", alpha=0.5)
axes[1].axvspan(0, 0.5, facecolor="gray", alpha=0.5)
plt.tight_layout()
plt.legend(loc="best", frameon=False, fontsize=11)
namepic="CV2_Pob_CR_%d_wnd_%.3f_stp_%.3f_Tac.png"%(CR, wnd, stp)
fig.savefig(namepic, dpi=300, bbox_inches='tight', transparent=True)


# Figura para el estímulo auditivo
fig, axes=plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
count=0
#ccolor=1
ccolorTac=0
ccolorAud=0
for i in Amplitudes:
    if i ==0 or i>24:    #Caso vibrotáctil
        print("Vamos en i: %.2f\t y count=%d  "%(i, count))
        axes[0].plot(t, savgol_filter( (Datos[keyCV_A1][count, :] )/(Datos[keyCVN_A1][count, :]), 7, 3) , c=colorAud[ccolorAud]) 
        axes[1].plot(t, savgol_filter( (Datos[keyCV_3b][count, :] )/(Datos[keyCVN_3b][count, :]), 7, 3), c=colorAud[ccolorAud], label="%d db"%(int(round(Amplitudes[count])))) 
        count+=1
        ccolorAud+=1
    else:
        count+=1
plt.show(block=False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_xlabel("Time [s]", fontsize=14)
axes[0].set_title("Neuronas del área 1", fontsize=14)
axes[1].set_title("Neuronas del área 3b", fontsize=14)
axes[0].axvspan(0, 0.5, facecolor="gray", alpha=0.5)
axes[1].axvspan(0, 0.5, facecolor="gray", alpha=0.5)
plt.tight_layout()
plt.legend(loc="best", frameon=False, fontsize=11)
namepic="CV2_Pob_CR_%d_wnd_%.3f_stp_%.3f_Aud.png"%(CR, wnd, stp)
fig.savefig(namepic, dpi=300, bbox_inches='tight', transparent=True)
