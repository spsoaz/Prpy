#Carga librerías
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
cmap=cm.get_cmap("tab20")
import Funciones_AjustesLinealesVsSigmoidales_PorNeourona as ALS
#-______-Parámetros del script
typedata=0  # Puede ser 3b :1 o A1:0 
hits=0     #0 indica aciertos y errores, 1 indica solo aciertos
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#----------------------------------          Define constantes para el manejo de archivos  y funciones  ----------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

if typedata==0:
    area="A1"
else:
    area="3b"
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
r=np.size(Tasas, axis=1)- headers["Data"]
t=np.arange(-0.95, -1.95+(r)*0.010, 0.010) 

ini=100 + headers["Data"]  #Posición encontrada a mano
fin=150 + headers["Data"]  #Encontrado a mano
array=list(np.arange(ini, fin, 5))
neuronas=np.concatenate((neuronas32, neuronas33), axis=0)
Segment=np.sum(Tasas[:, array], axis=1)   #Tasa que considera todo el estímulo
#Estandarizado de datos
Amplitudes=np.unique(Tasas[:, headers["Amp"]])
DatosAvg=np.zeros((np.size(neuronas, axis=0), 5 + np.size(Amplitudes) ))
DatosAvg[:, :]=np.NAN
SegmentN=np.zeros((np.size(DatosAvg, axis=0), np.size(Amplitudes)))
row=0
if hits==1:
    maskhits=Tasas[:, headers["Hits"]]==1
else:
    maskhits=np.ones((np.size(Tasas, axis=0)), dtype=bool)

for serie, elec, neu, CR, b3  in neuronas:        
    mask=(Tasas[:, headers["Serie"]]==serie)*(Tasas[:, headers["Elec"]]==elec)*(Tasas[:, headers["Uni"]]==neu)*(maskhits)
    mask2=Tasas[:, headers["Amp"]]==0  #Amplitud =0
    mask3=Tasas[:, headers["Amp"]]==24  #Amplitud =24
    media=np.mean(Segment[mask*mask2] )
    proof=np.mean(Segment[mask*mask3] )   
    sigma=np.std(Segment[mask*mask2] )
    DatosAvg[row, [0, 1, 2, 3, 4]]=[serie, elec, neu, CR, b3]
    if media<proof:       
        div=np.mean(Segment[mask*mask3] )
        #print(np.mean(Segment[mask*mask3] ), "\t", np.mean(Segment[mask*mask2]), "\t", div)
        col=0
        for amp in  Amplitudes:
            mask2=Tasas[:, headers["Amp"]]==amp 
            if np.sum(mask2)>0:
                DatosAvg[row, 5+ col]=np.mean(Segment[mask*mask2])/div
                if sigma>1e-3:
                    SegmentN[row, col]=np.mean( ((Segment[mask*mask2]-media))/( (sigma))  )
                else:
                     SegmentN[row, col]=np.mean( ((Segment[mask*mask2]-media)) )
            col+=1         
    row+=1        
       
del RR032_a, RR032_b, RR032_c, neuronas32, neuronas33, t, r, array, ini, fin,  maskZero,  mask, mask2, mask3, media, sigma, proof

# Cálculo del ajuste por tipo de campo receptor así como por tipo de área (3b o A1)    
mask3b=Tasas[:, headers["3b"]]==1
TacAmp=np.unique(Tasas[maskTac, headers["Amp"]])
AudAmp=np.unique(Tasas[maskAud, headers["Amp"]])

maskCR1_total=DatosAvg[:, 3]==1
maskCR2_total=DatosAvg[:, 3]==2
maskCR3_total=DatosAvg[:, 3]>2
mask3b_total=DatosAvg[:, 4]==typedata
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  Gráficas - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Gráfica de CR1 A1
TacAmplog=np.concatenate(([0], np.log10(TacAmp[1::])))  
mediaTac=np.zeros_like(TacAmp)
stdTac=np.zeros_like(TacAmp)
mediaTac2=np.zeros_like(TacAmp)
stdTac2=np.zeros_like(TacAmp)
mediaAud=np.zeros_like(AudAmp)
stdAud=np.zeros_like(AudAmp)
mediaAud2=np.zeros_like(AudAmp)
stdAud2=np.zeros_like(AudAmp)

# Método que incluye el promedio por neurona
mediaTacTotal=np.zeros_like(TacAmp)
stdTacTotal=np.zeros_like(TacAmp)
mediaAudTotal=np.zeros_like(AudAmp)
stdAudTotal=np.zeros_like(AudAmp)
maskTac2=np.where(Amplitudes<27)[0]
maskAud2=np.where((Amplitudes==0) + (Amplitudes>27))[0]

Ampmatrix=np.zeros((np.sum(maskCR1_total*(~mask3b_total)), np.size(DatosAvg, axis=1)-5 ))
for amp in  range(len(maskTac2)):   
    mediaTac[amp]=np.nanmean(DatosAvg[maskCR1_total*(~mask3b_total), maskTac2[amp] + 5])
    Ampmatrix[:, amp]=Amplitudes[maskTac2[amp]]
    stdTac[amp]=np.nanstd(DatosAvg[maskCR1_total*(~mask3b_total), maskTac2[amp] + 5])

#for amp in  range(len(maskAud2)):
#   mediaAud[amp]=np.nanmean(DatosAvg[maskCR1_total, maskAud2[amp] + 5])
#   stdAud[amp]=np.nanstd(DatosAvg[maskCR1_total, maskAud2[amp] + 5])
    
x=Ampmatrix.flatten()
y=(DatosAvg[maskCR1_total*(~mask3b_total), 5::]).flatten()
mask=(np.isnan(y)) + (x==0)
y=y[~mask]
x=x[~mask]
x=np.log10(x)
del mask
fig, ax=plt.subplots(1, 1, figsize=(12, 9))
ax.scatter(TacAmplog[1::], mediaTac[1::], c="blue", label="CR1 " + area)
ax.errorbar(TacAmplog[1::], mediaTac[1::], stdTac[1::], c="blue", fmt=' ')
modelin, covs=curve_fit(ALS.linear, xdata=x, ydata=y)
xm=np.linspace(np.min(x), np.max(x), 200)
ax.plot(xm, modelin[0]*xm +modelin[1], c="blue", alpha=0.3, lw=3)
def sigmoidal(x, a, b, c ,d):
    return (a-b)/(1 + np.exp((d-x)/c) ) + b
model, covs=curve_fit(sigmoidal, xdata=x, ydata=y)
ax.plot(xm, sigmoidal(xm, model[0], model[1], model[2], model[3]), c="blue", alpha=0.5, lw=3)

ampt=np.unique(Ampmatrix)
labs=[str(int(ampt[ i])) for i in range(1, len(ampt))]
plt.xticks(ticks=np.log10(ampt[1::]), labels=labs, fontsize=22)
plt.yticks(fontsize=22)
ax.legend(loc="best", frameon=False, fontsize=22)
# Calculo del AIC
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#Ahora para el campo receptor 2
mediaTacTotalCR2=np.zeros_like(TacAmp)
stdTacTotalCR2=np.zeros_like(TacAmp)
mediaAudTotalCR2=np.zeros_like(AudAmp)
stdAudTotalCR2=np.zeros_like(AudAmp)
maskTac2=np.where(Amplitudes<27)[0]
maskAud2=np.where((Amplitudes==0) + (Amplitudes>27))[0]

Ampmatrix=np.zeros((np.sum(maskCR2_total*(~mask3b_total)), np.size(DatosAvg, axis=1)-5 ))
for amp in  range(len(maskTac2)):   
    mediaTac[amp]=np.nanmean(DatosAvg[maskCR2_total*(~mask3b_total), maskTac2[amp] + 5])
    Ampmatrix[:, amp]=Amplitudes[maskTac2[amp]]
    stdTac[amp]=np.nanstd(DatosAvg[maskCR2_total*(~mask3b_total), maskTac2[amp] + 5])

#for amp in  range(len(maskAud2)):
#    mediaAud[amp]=np.nanmean(DatosAvg[maskCR2_total, maskAud2[amp] + 5])
#    stdAud[amp]=np.nanstd(DatosAvg[maskCR2_total, maskAud2[amp] + 5])
    
x=Ampmatrix.flatten()
y=(DatosAvg[maskCR2_total*(~mask3b_total), 5::]).flatten()
mask=(np.isnan(y)) + (x==0)
y=y[~mask]
x=x[~mask]
x=np.log10(x)
del mask
ax.scatter(TacAmplog[1::], mediaTac[1::], c="red", label="CR2 " + area)
ax.errorbar(TacAmplog[1::], mediaTac[1::], stdTac[1::], c="red", fmt=' ')
modelin, covs=curve_fit(ALS.linear, xdata=x, ydata=y)
xm=np.linspace(np.min(x), np.max(x), 200)
ax.plot(xm, modelin[0]*xm +modelin[1], c="red", alpha=0.3, lw=4)

model, covs=curve_fit(sigmoidal, xdata=x, ydata=y)
ax.plot(xm, sigmoidal(xm, model[0], model[1], model[2], model[3]), c="red", alpha=0.5, lw=4)
ax.legend(loc="best", frameon=False, fontsize=22)

#Ahora para el campo receptor 3
mediaTacTotalCR3=np.zeros_like(TacAmp)
stdTacTotalCR3=np.zeros_like(TacAmp)
mediaAudTotalCR3=np.zeros_like(AudAmp)
stdAudTotalCR3=np.zeros_like(AudAmp)
maskTac2=np.where(Amplitudes<27)[0]
maskAud2=np.where((Amplitudes==0) + (Amplitudes>27))[0]

Ampmatrix=np.zeros((np.sum(maskCR3_total*(~mask3b_total)), np.size(DatosAvg, axis=1)-5 ))
for amp in  range(len(maskTac2)):   
    mediaTac[amp]=np.nanmean(DatosAvg[maskCR3_total*(~mask3b_total), maskTac2[amp] + 5])
    Ampmatrix[:, amp]=Amplitudes[maskTac2[amp]]
    stdTac[amp]=np.nanstd(DatosAvg[maskCR2_total*(~mask3b_total), maskTac2[amp] + 5])

#for amp in  range(len(maskAud2)):
#    mediaAud[amp]=np.nanmean(DatosAvg[maskCR2_total, maskAud2[amp] + 5])
#    stdAud[amp]=np.nanstd(DatosAvg[maskCR2_total, maskAud2[amp] + 5])
    
x=Ampmatrix.flatten()
y=(DatosAvg[maskCR3_total*(~mask3b_total), 5::]).flatten()
mask=(np.isnan(y)) + (x==0)
y=y[~mask]
x=x[~mask]
x=np.log10(x)
del mask
ax.scatter(TacAmplog[1::], mediaTac[1::], c="green", label="CR3 " + area)
ax.errorbar(TacAmplog[1::], mediaTac[1::], stdTac[1::], c="green", fmt=' ')
modelin, covs=curve_fit(ALS.linear, xdata=x, ydata=y)
xm=np.linspace(np.min(x), np.max(x), 200)
ax.plot(xm, modelin[0]*xm +modelin[1], c="green", alpha=0.3, lw=4)

model, covs=curve_fit(sigmoidal, xdata=x, ydata=y)
ax.plot(xm, sigmoidal(xm, model[0], model[1], model[2], model[3]), c="green", alpha=0.5, lw=4)
ax.legend(loc="best", frameon=False, fontsize=22)
"""
div=np.nanmean(SegmentN[:, 9])
for amp in  range(len(maskTac2)):   
    mediaTac2[amp]=np.nanmean(SegmentN[:, maskTac2[amp] ])/div
    stdTac2[amp]=np.nanstd(SegmentN[:, maskTac2[amp] ]/div)

for amp in  range(len(maskAud2)):
    mediaAud2[amp]=np.nanmean(SegmentN[:, maskAud2[amp] ])
    stdAud2[amp]=np.nanstd(SegmentN[:, maskAud2[amp]])
    
plt.figure()
plt.scatter(TacAmp[1::], mediaTac2[1::], c="black")
plt.errorbar(TacAmp[1::], mediaTac2[1::], stdTac2[1::], c="black")
plt.plot(TacAmp[1::], mediaTac2[1::], c="black")
plt.title("Estímulo vibrotáctil, CR=1,   A1 Método 1")
plt.show(block=False)
"""
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Amplitude [$\mu$m]", fontsize=24)
ax.set_ylabel("Normalized activity", fontsize=24)
plt.tight_layout()
filename=input("Coloque el nombre del archivo junto con la dirección /.. \n")
print("La figura se guardará con nombre: ", filename)
plt.savefig(filename, format="svg")
plt.show(block=False)
