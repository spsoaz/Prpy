#Carga librerías
import numpy as np
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

#-______-Parámetros del script
wnd_end=0.3
stp=0.010

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
R33=np.loadtxt(root+"/RR0330Tasa_individual.csv", delimiter=',')
RR032_a=np.loadtxt(root+"/RR032Tasa_individual_RR032P1.csv", delimiter=',')
RR032_b=np.loadtxt(root+"/RR032Tasa_individual_RR032P2.csv", delimiter=',')
RR032_c=np.loadtxt(root+"/RR032Tasa_individual_RR032P3.csv", delimiter=',')
R32=np.concatenate((RR032_a, RR032_b, RR032_c), axis=0)
neuronas32=np.unique(R32[:, [headers["Serie"], headers["Elec"], headers["Uni"], headers["CR"], headers["3b"] ]], axis=0)
neuronas33=np.unique(R33[:, [headers["Serie"], headers["Elec"], headers["Uni"], headers["CR"], headers["3b"] ]], axis=0)
Tasas=np.concatenate((R32, R33), axis=0)
maskTac=(Tasas[:, headers["Amp"]]==0) + (Tasas[:, headers["Amp"]]>3)
maskAud=Tasas[:, headers["Amp"]]<3
maskZero=(Tasas[:, headers["Amp"]]==0)
Tasas[(maskAud.astype(int)- maskZero.astype(int) ).astype(bool), headers["Amp"]]=ALS.fitmatlab(Tasas[(maskAud.astype(int)- maskZero.astype(int) ).astype(bool), headers["Amp"]]  )
Tasas=ALS.Transform_Data(headers["Data"], Tasas, 50, wnd_end*1000)   #Adapatación de las tasas(conteo) a la venatana de interés.
Amplitudes=np.unique(Tasas[:, headers["Amp"]])
neuronas=np.concatenate((neuronas32, neuronas33), axis=0)
Nneu=np.size(neuronas, axis=0)
del RR032_a, RR032_b, RR032_c, R32, R33, neuronas32, neuronas33, maskTac, maskAud, maskZero  #Destruct objects

#Tasas=ALS.Transform_Data(headers["Data"], Tasas, 50, wnd_end)   
finaliza=np.size(Tasas, axis=1)
cols=headers["Data"]
#-------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------Cuerpo de los cálculos     ----------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
varianzas=np.zeros((Nneu, np.size(Amplitudes), finaliza-cols))  #Arreglo de vrianzas, 
varianzas[:, :, :]=np.NAN
medias=np.copy(varianzas)                                                     #Arreglo de medias, NAN es la etiqueta de no escritura.
row=0
for serie, elec, unit, _, _  in neuronas:
    #print("Neurona:     %f \t %f \t %f"%(serie, elec, unit))
    maskneu=(Tasas[:, headers["Serie"] ]==serie)*(Tasas[:, headers["Elec"] ]==elec)*(Tasas[:, headers["Uni"] ]==unit)
    colss=0    #Inicio de la variable que va sobre el tiempo
    while(cols + colss< finaliza):     #Ciclo que va sobre el tiempo,
        submatriz=Tasas[maskneu, cols+colss]
        amp_index=0
        for amp in Amplitudes:            
            tmp=Tasas[maskneu, headers["Amp"]]==amp
            if np.sum(tmp)>0:
                medias[row, amp_index, colss]=np.nanmean(submatriz[tmp])
                varianzas[row, amp_index, colss]=np.nanvar(submatriz[tmp])
            amp_index+=1
        colss+=1
    row+=1
# Cálculo del ajuste por tipo de campo receptor así como por tipo de área (3b o A1)    
maskCR1=neuronas[:, 3]==1
maskCR2=neuronas[:, 3]==2
maskCR3=neuronas[:,3]>2
mask3b=neuronas[:, 4]==1
maskTac=Amplitudes<27
maskAud=(~maskTac) + (Amplitudes==0)
# Fano de los distintos campos receptores en A1 y 3b
FanoCR1A1=np.zeros((2, finaliza-cols))
FanoCR13b=np.copy(FanoCR1A1)
FanoCR2A1=np.copy(FanoCR1A1)
FanoCR23b=np.copy(FanoCR1A1)
FanoCR3A1=np.copy(FanoCR1A1)
FanoCR33b=np.copy(FanoCR1A1)
#Cálculo del Factor de Fano
for t in range(finaliza-cols):
    # CR1  3b
    mediastmp=(medias[maskCR1*mask3b, :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR1*mask3b, :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params,  _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR13b[0, t]=params[0]
    mediastmp=(medias[maskCR1*mask3b, :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR1*mask3b, :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR13b[1, t]=params[0]
    # CR1 A1
    mediastmp=(medias[maskCR1*(~mask3b), :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR1*(~mask3b), :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp])
    FanoCR1A1[0, t]=params[0]
    mediastmp=(medias[maskCR1*(~mask3b), :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR1*(~mask3b), :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR1A1[1, t]=params[0]
    # CR2 3b
    mediastmp=(medias[maskCR2*mask3b, :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR2*mask3b, :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR23b[0, t]=params[0]
    mediastmp=(medias[maskCR2*mask3b, :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR2*mask3b, :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp])
    FanoCR23b[1, t]=params[0]
    # CR2 A1
    mediastmp=(medias[maskCR2*(~mask3b), :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR2*(~mask3b), :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params , _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR2A1[0, t]=params[0]
    mediastmp=(medias[maskCR2*(~mask3b), :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR2*(~mask3b), :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR2A1[1, t]=params[0]
    # CR3 3b
    mediastmp=(medias[maskCR3*mask3b, :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR3*mask3b, :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR33b[0, t]=params[0]
    mediastmp=(medias[maskCR3*mask3b, :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR3*mask3b, :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR33b[1, t]=params[0]
    # CR3 A1
    mediastmp=(medias[maskCR3*(~mask3b), :, t])[:, maskTac]
    varianzastmp=(varianzas[maskCR3*(~mask3b), :, t])[:, maskTac]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )
    FanoCR3A1[0, t]=params[0]
    mediastmp=(medias[maskCR3*(~mask3b), :, t])[:, maskAud]
    varianzastmp=(varianzas[maskCR3*(~mask3b), :, t])[:, maskAud]
    msktmp=(~np.isnan(mediastmp))*( ~np.isnan(varianzastmp))
    params, _=curve_fit(ALS.linear, xdata=mediastmp[msktmp] ,  ydata=varianzastmp[msktmp] )    
    FanoCR3A1[1, t]=params[0]
#Guardado de archivos
t=np.arange(-2 + wnd_end,  -2+wnd_end+(finaliza-cols)*0.010, 0.010)
plt.plot(t, FanoCR3A1[0, :])


#strfile="Fano_Factor_Pob_32_33_wnd%.3f_stp10.pkl"%(wnd_end)
#filea=open(strfile, "wb")
#R={"t":t, "wnd":wnd_end, "stp":10, "FanoCR1A1":FanoCR1A1,"FanoCR13b":FanoCR13b, "FanoCR2A1":FanoCR2A1, "FanoCR23b":FanoCR23b,"FanoCR3A1":FanoCR3A1,"FanoCR33b":FanoCR33b,"pythonversion":"3.8.6", "caracteristicas":"El primer renglón es táctil, el segundo auditivo"}
#pkl.dump(R, filea)
#filea.close()

#strfile="Fano_Factor_Pob_32_33_wnd%.3f_stp10.json"%(wnd_end)
#filea=open(strfile, "w")
#R={"t":list(t), "wnd":wnd_end, "stp":10, "FanoCR1A1":tuple(FanoCR1A1.tolist()),"FanoCR13b":tuple(FanoCR13b.tolist()), "FanoCR2A1":tuple(FanoCR2A1.tolist()), "FanoCR23b":tuple(FanoCR23b.tolist()),"FanoCR3A1":tuple(FanoCR3A1.tolist()),"FanoCR33b":tuple(FanoCR33b.tolist()),"pythonversion":"3.8.6", "caracteristicas":"El primer renglón es táctil, el segundo auditivo"}


#json.dump(R, filea)
#filea.close()

# Prueba para respaldo previo al c�lculo del Fano
strfile="Fano_Factor_Pob_Origins_wnd%.3f_stp10b.pkl"%(wnd_end)
filea=open(strfile, "wb")
R={"t":t, "wnd":wnd_end, "stp":10, "var":varianzas,"mean":medias, "neu_data":neuronas, "Tasas":Tasas,"amps_order":Amplitudes,"headers_Tasas":headers,"pythonversion":"3.8.6", "headers_neu_data":{"Serie":0,  "Elec":1, "Uni":2,  "CR":3,  "3b":4}}
pkl.dump(R, filea)
filea.close()
