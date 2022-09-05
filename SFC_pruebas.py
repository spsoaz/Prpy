from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from skimage.measure import label, regionprops_table
from numba import njit
from os import name as nameos
from sys import path, getsizeof
import matplotlib as mpl
import matplotlib.style as ms
mpl.rcParams.update({'font.size': 8, 'font.weight':'bold', 'font.style':'normal'})
plt.rc('font', family='serif')
mpl.rc('font', size=8)  # change font size from default 10
ms.use('default')
plt.rcParams['svg.fonttype'] = 'none'


if nameos=="posix":
    path.append("/home/sparra/MEGA/Prpy")
else:
    path.append("C:\\Users\\sparra\\Documents\\Prpy")
# import Phd_extV2 as Pext

# from Plots import beauty, cm
import lfpy as lfp
from lfpy import AdapWinSpec as lfp_AdapWinSpec
import lfpyclass as lfpycls
reload(lfp)
reload(lfpycls)

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
data={
        "monkey":33,                           # Mon, solo 32 o 33
        "serie":77,                            # Serie o día de registro
        "orden": 1,                            # Orden del set. 2 es Unc en RR032
        "electrodes":(1,2,3,4,5,6,7),          # Electrodos válidos en el estudio.
        "area":"S1",                           # Area de registro
        "subarea":"3b",                        # Sub área de registro        
        "lti":1.5,                             # Tiempo de corte hacia atrás
        "dur":3,                               # Duration of the segment to take in sec.
        #nyquist=1000                          # Frecuencia de Nyquist
        "fsamp":2000,                          # Frecuencia de muestreo 
        "computador":1,                        # computador en que se realizarán los cálculos 3 lab 1 lnvo
        "neurons":((1,), (1, ), (1, ), (1, 2), (1, 3), (), (2, )),   # Unidades de neuronas
}
reg1=lfpycls.lfps_ps(**data) # Creamos objeto de clase
reg1.loadspks()            # Cargamos espigas asociadas a este registro.
reg1.loadsignal(data["computador"], fun=(0,))  # Cargamos señales con filtro de remoción de ruido (notch)
reg1_rrarray=reg1.dict2ndarray()
freqs=np.logspace(np.log10(4), np.log10(130), 40)   # freqs para la FFT
reg1_rrarray[:, 23::]=reg1_rrarray[:, 23::]-np.mean(reg1_rrarray[:, 23::], axis=2)[:, :, None] # rreferencing common average
r,c=40, 9000    # valores extraídos de las pruebas.
reg1_rrarrayX=[np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               np.zeros((r, c, len(reg1_rrarray))),
               ]
for elec_i in range(len(data["electrodes"])):
    for trial_i in range(reg1_rrarray.shape[0]):
        if trial_i==0 and elec_i==0:
            # time, _, reg1_rrarrayX[elec_i][:, :, trial_i]=lfp_AdapWinSpec(reg1_rrarray[trial_i, 23::, elec_i], reg1.fsamp, 0.85,
            #                        Ncicles=5,  freqs=freqs,  window=("multitaper", 4))
            time, _, reg1_rrarrayX[elec_i][:, :, trial_i]=lfp.MorletWaveletSpectrogram(reg1_rrarray[trial_i, 23::, elec_i], 
                                                              reg1.fsamp, Ncicles=5, freqs=freqs, lentimewav=5)
        else:
            # _, _, reg1_rrarrayX[elec_i][:, :, trial_i]=lfp_AdapWinSpec(reg1_rrarray[trial_i, 23::, elec_i], reg1.fsamp, 0.85,
            #                        Ncicles=5,  freqs=freqs,  window=("multitaper", 4))
            _, _, reg1_rrarrayX[elec_i][:, :, trial_i]=lfp.MorletWaveletSpectrogram(reg1_rrarray[trial_i, 23::, elec_i], 
                                                              reg1.fsamp, Ncicles=5, freqs=freqs, lentimewav=5)

    print("Electrodo %d finalizado"%(elec_i))


# Graficación según las intensidades 
trials=(reg1_rrarray[:, 0, 0]-1).astype(np.int8)
maskZ=np.where(np.logical_and(reg1.psyc[trials, 11]==0, reg1.psyc[trials, 13]==0))[0]
maskTsupT=np.where(np.logical_and(reg1.psyc[trials, 11]>10, reg1.psyc[trials, 13]==0))[0]
# maskTsubT=np.where(np.logical_and(reg1.psyc[trials, 11]<7, reg1.psyc[trials, 13]==0))[0]
# maskTumbT=np.where(np.logical_and(np.logical_and(reg1.psyc[trials, 11]<7, reg1.psyc[trials, 13]==0),
#                     reg1.psyc[trials, 11]> 4 ))[0]
# maskTsupA=np.where(np.logical_and(reg1.psyc[trials, 11]==0, reg1.psyc[trials, 13]> ))[0]
# maskTsubA=np.where(np.logical_and(reg1.psyc[trials, 11]==0, reg1.psyc[trials, 13]< ))[0]
# maskTumbA=maskTumbT=np.where(np.logical_and(np.logical_and(reg1.psyc[trials, 13]<7, reg1.psyc[trials, 11]==0),
                    # reg1.psyc[trials, 13]>   ))[0]

for elec_i in range(len(data["electrodes"])):
    fig, ax=plt.subplots(2, 1, figsize=(4, 8), sharex=True, sharey=True)
    xfig=ax[0].imshow(reg1_rrarrayX[elec_i][:, :, maskZ].mean(axis=2), 
                      aspect="auto", origin="lower", 
                      extent=(time[0], time[-1], freqs[0], freqs[-1]),
                      vmin=np.min(reg1_rrarrayX), vmax=np.max(reg1_rrarrayX))
    # prueba1=reg1_rrarrayX[elec_i][:, 0:1800, maskZ]
    # prueba2=np.reshape(prueba1, (r, 1800*len(maskZ)))
    plt.colorbar(xfig, ax=ax[0] ) # cax=
    norm=np.zeros_like(reg1_rrarrayX[elec_i][:, :, 0])
    for trial_i in maskZ:
        norm+=lfp.normalize(reg1_rrarrayX[elec_i][:, :, trial_i], 2000, segment=reg1_rrarrayX[elec_i][:, 0:1800, trial_i])
    
    
    norm/=len(maskZ)
    xfig2=ax[1].imshow(norm, aspect="auto", origin="lower", 
                      extent=(time[0], time[-1], freqs[0], freqs[-1]),
                      )
    plt.colorbar(xfig2, ax=ax[1]) #, cax=ax[1]
    for ax_i in ax.flatten():
        ax_i.set_xlabel("Time [s]", fontsize=8)
        ax_i.set_ylabel("Freqs. [Hz]", fontsize=8)
    
    
    
    
    plt.tight_layout()
    plt.show()
    # Now normalize the scalogram.


import dill                            #pip install dill --user
filename = 'sfc_pruebas.pkl'
dill.dump_session(filename)


