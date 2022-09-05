from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import lfpy as lfp
import os
import pickle as pkl
reload(lfp)
from skimage.measure import label, regionprops_table
from numba import njit
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

class InputParameters:
    def __init__(self, monkey, serie, orden, electrodes, nperm, pval):
        self.m=monkey
        self.s=serie
        self.o=orden
        self.e=electrodes
        self.nperm=nperm
        self.pval=pval
        return None
    
@njit()
def itpc (Phase, n):
    return np.abs(np.sum(np.exp(1j*Phase), axis=2)/n)**2    

@njit()
def Permut(Grupo1, Grupo2,  n1, n2, nperm, matrixperm, ITPC_perm1, ITPC_perm2):
    """
    This function only calculates the permutations by shuffling data of two conditions.
    Parameters
    ----------
    Grupo1 : TYPE
        DESCRIPTION.
    Grupo2 : TYPE
        DESCRIPTION.
    n1 : TYPE
        DESCRIPTION.
    n2 : TYPE
        DESCRIPTION.
    nperm : TYPE
        DESCRIPTION.
    matrixperm : TYPE
        DESCRIPTION.
    ITPC_perm1 : TYPE
        DESCRIPTION.
    ITPC_perm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    Juntas=np.concatenate((Grupo1, Grupo2), axis=2)
    for perm_i in range(nperm):
        Grupo1_New=Juntas[:, :, matrixperm[0:n1, perm_i]]
        Grupo2_New=Juntas[:, :, matrixperm[n1::, perm_i]]
        ITPC_perm1[:, :, perm_i]=itpc(Grupo1_New, n1)
        ITPC_perm2[:, :, perm_i]=itpc(Grupo2_New, n2)
    return None

@njit()
def ITPCPermut(Matrix, r, c, ntrials, nperm, ITPC_perm, indices):
    """
    Esta función calculará las permutaciones para la identificaión de un valor
    de ITPC significativo per sé.
    Parameters
    ----------
    Matrix : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    ntrials : TYPE
        DESCRIPTION.
    nperm : TYPE
        DESCRIPTION.
    ITPC_perm : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for perm_i in range(nperm):
        Datatmp=np.zeros((r, c, ntrials), dtype=np.float32)
        for trial_i in range(ntrials):   
            for c_i in range(c):
                Datatmp[:, c_i, trial_i]=Matrix[:, indices[trial_i*perm_i + trial_i, c_i], trial_i]
        ITPC_perm[:, :, perm_i]=itpc(Datatmp, ntrials)
    return None

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(monkey, serie, orden, electrode, computador, nperm=1000, pval=0.01, **kwargs):
    """
    Función principal, esta función realiza el cálculo del ITPC (Inter Trial Phase Consistency) y guarda los resultados.
    Además, esta función prepara los datos para las evaluaciones de significancia.
    
    Esta función busca directamente en los archivos del mono.
    
    En el caso de la evaluación de significancia per sé, esta funcion utiliza el método de la 
    comparación con el ITPC_crítico con el fin de establecer la umbralización. Este ITPC_crit es obtenido
    con base en eel srtículo de Zar (1999) (ver Cohen eq. 34.2 para mayores referencias.)
    
    Esta función tiene las siguientes parámetros de entrada:
    
    1.- monkey  : Int  Id del mono a trabajar, 32 o 33.
    2.- serie   : Int Serie de registro, día de registro, entero. 187, 152 p. ej
    3.- orden   : Int Orden de registro, 1, 2, 3... 2 es el default de incertidumbre para el mono 32
    4.- electrode:  (Int) El electrodo del cual se leerá el archivo
    5.- computador: (Int) el compútador destinado (de este dependen las direcciones para la carga correcta de datos.)
    6.- nperm:   (Int) número de permutaciones a realizar para el cálculo de significancia, default=1000
    7.- pval: (float) Significancia seg+un la cual se decidirá si un pixel o cluster de pixeles es significativo o no. Default 0.01
    
    """  
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
    # Cargar espectros precalculados con anteriordidad
    namefile="m%d_s%d_o%d_e%d_Spectrum.pkl"%(monkey, serie, orden, electrode)
    f=open(os.path.join(alternativepath, namefile), "rb")
    Data=pkl.load(f)
    f.close()
    Psych=Data["Psychophysics"]
    SenalesPhase=Data["Phase"][0:-2,:]
    nbands, lent, ntrials=SenalesPhase.shape
    band=Data["banda"][0:-2]
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
    mascaras=[np.where(Psych[:, P["AAud3"]]>=Thresholding[1, 1])[0],
              np.where(Psych[:, P["ATac3"]]>=Thresholding[0, 1])[0],
              np.where((Psych[:, P["AAud3"]]>Thresholding[1, 0])*(Psych[:, P["AAud3"]]<Thresholding[1, 1]))[0],
              np.where((Psych[:, P["ATac3"]]>Thresholding[0, 0])*(Psych[:, P["ATac3"]]<Thresholding[0, 1])  )[0],
              np.where((Psych[:, P["AAud3"]]<=Thresholding[1, 0])*(Psych[:, P["AAud3"]]>0)  )[0],
              np.where((Psych[:, P["ATac3"]]<=Thresholding[0, 0])*(Psych[:, P["ATac3"]]>0)  )[0],
              np.where((Psych[:, P["ATac3"]]==0)*(Psych[:, P["AAud3"]]==0))[0] ]
    ordenlabel=( "Acoustic Suprathreshold",
     "Tactile Suprathreshold",
     "Acoustic Threshold",
     "Tactile Threshold",
     "Acoustic Subthreshold",
     "Tactile Subthreshold",
     "No stimulus")
    
    # Guardar archivos con los cálculos de la fase por banda, por ensayo y por electrodo. Archivos csv. Separación por commas.
    ITPC={"Information":{"Monkey": monkey, "serie":serie, "orden":orden, "electrode":electrode,
                         "FrequencyBands": band, "number":tuple(len(mascaras[i]) for i in range(len(mascaras)))}}
    for masc_i in range(len(ordenlabel)):        
        ITPC_tmp=itpc(SenalesPhase[:, :, mascaras[masc_i]], ITPC["Information"]["number"][masc_i])
        ITPC.update({ordenlabel[masc_i]: np.copy(ITPC_tmp)})
    ## Ahora viene la etapa de graficación para comprobar resultado
    for masc_i in range(len(ordenlabel)):        
        fig, ax=plt.subplots(1,1)
        cinfo=ax.imshow(ITPC[ordenlabel[masc_i]], origin="lower", aspect="auto", extent=[-1.5, 3, 1, nbands+1], vmin=0, vmax=1)
        im2=plt.colorbar(cinfo)
        for t in im2.ax.get_yticklabels():
                    t.set_fontsize(12)
        im2.set_label("ITPC", fontsize=12)
        ax.set_title(ordenlabel[masc_i], fontsize=14)
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Frequencies [Hz]", fontsize=12)
        ax.plot((0, 0, 0.5, 0.5),(1, nbands, nbands, 1), c="white", lw=1, ls="--")
        ax.set_ylim([1, nbands])
        ax.tick_params(labelsize=12)
        ax.set_yticks(np.arange(1, len(band), 3))
        ax.set_yticklabels(str(i) for i in band[1::3] )
        fig.tight_layout()
        namefig="ITPC_m%d_s%d_o%d_e%d_%s.svg"%(monkey, serie, orden, electrode, ordenlabel[masc_i])
        fig.savefig(os.path.join(alternativepath, namefig ))
        plt.close(fig)
        
    # Guardado de archivo
    namefile="ITPC_m%d_s%d_o%d_e%d.pkl"%(monkey, serie, orden, electrode)
    f=open(os.path.join(alternativepath, namefile), "wb")
    pkl.dump(ITPC, f)
    f.close()
    # Preparación del archivo con la signficancia
    namefile="ITPC_Significance_m%d_s%d_o%d_e%d.pkl"%(monkey, serie, orden, electrode)
    f=open(os.path.join(alternativepath, namefile), "wb")
    ITPC_significance={"Information":{"Monkey": monkey, "serie":serie, "orden":orden, "electrode":electrode,
                         "FrequencyBands": band, "number":tuple(len(mascaras[i]) for i in range(len(mascaras)))}}
    ### Ahora viene el cálculo de significancia   
    ## Cálculo de la significancia del ITPC (comparación contra nostimulus)
    n_0=ITPC["Information"]["number"][6]   # Número de ensayos con amplitud cero
    ITPC_perm1=np.zeros((nbands, lent, nperm), dtype=np.float32)
    ITPC_perm2=np.zeros((nbands, lent, nperm), dtype=np.float32)
    indicesP=np.zeros((55*nperm, lent), dtype=np.int16)
    for i in range(55*nperm):        
        cutpoint=np.random.randint(1, lent-1, 1)
        indicesP[i, :]=np.concatenate((np.arange(cutpoint, lent),  np.arange(cutpoint)))
    for  masc_i in range(len(ordenlabel)):
        # if masc_i==1:  ## Estamos en Tactile supra
        #     print("Ya llegué")
        #     pass
        n_i=ITPC["Information"]["number"][masc_i]  
        ITPC_sig=np.zeros((nbands, lent, nperm), dtype=np.float32)
        matrixperm=np.repeat(np.arange(n_0 + n_i, dtype=np.int16).reshape(n_0+n_i, 1), nperm, axis=1)
        for perm_i in range(nperm):
            np.random.shuffle(matrixperm[:, perm_i])
        Permut(SenalesPhase[:, :, mascaras[masc_i]], SenalesPhase[:, :, mascaras[6]],
               n_i, n_0, nperm, matrixperm, ITPC_perm1, ITPC_perm2)
        diferencias_or=np.abs(ITPC[ordenlabel[masc_i]]-ITPC[ordenlabel[6]])
        diferencias_perm=np.abs(ITPC_perm1-ITPC_perm2)
        diferencias_permsort=np.copy(diferencias_perm)
        diferencias_permsort.sort(axis=2)
        thresholds=diferencias_permsort[:, :, np.int32((1-pval)*nperm)]    
        del diferencias_permsort
        Umbralizada=diferencias_or>=thresholds
        label_Umbralizada, countslabels=label(Umbralizada, return_num=True)
        properties=regionprops_table(label_Umbralizada, intensity_image=diferencias_or, properties=('mean_intensity', "area", "label"), cache=True)
        # Ahora el cálculo de las regiones por cada permutación
        sizes=np.zeros((2, nperm), dtype=np.int32)
        for perm_i in range(nperm):
            Umbral_perm=diferencias_perm[:, :, perm_i]>=thresholds
            label_perm=label(Umbral_perm)
            properties_perm=regionprops_table(label_perm, intensity_image=diferencias_perm[:, :, perm_i], properties=('mean_intensity', "area"), cache=True)
            sizes[0, perm_i]=np.max(properties_perm["mean_intensity"]*properties_perm["area"])
            sizes[1, perm_i]=np.max(properties_perm["area"])
        # Ahora a extraer el valor 
        sizes.sort()
        ThresholdSize=sizes[:, np.int32(nperm*(1-pval))]
        ## AHora viene la corrección por comparaciones m+ultiples según la corrección por agrupamiento y su suma máxima.
        maskArea=properties["mean_intensity"]*properties["area"]>=ThresholdSize[0]  
        if not maskArea.any():   # Nada pasa la corrección
            ClusCorr=np.zeros((nbands, lent), dtype=bool)
        else:                   # Algunos la pasan  
            indices=np.where(maskArea)[0]
            ClusCorr=np.zeros((nbands, lent), dtype=bool)
            for label_i in range(len(indices)):
                masktmp=label_Umbralizada==properties["label"][indices[label_i]]
                ClusCorr += masktmp
        # Guardamos una copia para el guardado final        
           
        ## Aquí ahora calculamos la significancia del ITPC per sé
        ITPCPermut(SenalesPhase[:, :, mascaras[masc_i]], nbands, lent, n_i, nperm, ITPC_sig, indicesP)
        # Ahora viene la umbralización
        ITPCPermutsort=np.copy(ITPC_sig)
        ITPCPermutsort.sort(axis=2)
        thresholds=ITPCPermutsort[:, :, np.int32((1-pval)*nperm)]    
        del ITPCPermutsort
        Umbralizada_sig=ITPC[ordenlabel[masc_i]]>=thresholds
        label_Umbralizada, countslabels=label(Umbralizada_sig, return_num=True)
        properties=regionprops_table(label_Umbralizada, intensity_image=ITPC[ordenlabel[masc_i]], properties=('mean_intensity', "area", "label"), cache=True)
        # Ahora el cálculo de las regiones por cada permutación
        sizes=np.zeros((2, nperm), dtype=np.int32)
        for perm_i in range(nperm):
            Umbral_perm=ITPC_sig[:, :, perm_i]>=thresholds
            #plt.figure()
            #plt.imshow(ITPC_sig[:, :, perm_i], aspect="auto", origin="lower")
            label_perm=label(Umbral_perm)
            properties_perm=regionprops_table(label_perm, intensity_image=ITPC_sig[:, :, perm_i], properties=('mean_intensity', "area"), cache=True)
            sizes[0, perm_i]=np.max(properties_perm["mean_intensity"]*properties_perm["area"])
            sizes[1, perm_i]=np.max(properties_perm["area"])
        # Ahora a extraer el valor 
        sizes.sort()
        ThresholdSize=sizes[:, np.int32(nperm*(1-pval))]
        ## AHora viene la corrección por comparaciones m+ultiples según la corrección por agrupamiento y su suma máxima.
        maskArea=properties["mean_intensity"]*properties["area"]>=ThresholdSize[0]  
        if not maskArea.any():   # Nada pasa la corrección
            ClusCorr_sig=np.zeros((nbands, lent), dtype=bool)
        else:                   # Algunos la pasan            
            indices=np.where(maskArea)[0]
            ClusCorr_sig=np.zeros((nbands, lent), dtype=bool)
            for label_i in range(len(indices)):
                masktmp=label_Umbralizada==properties["label"][indices[label_i]]
                ClusCorr_sig += masktmp
        
        # Generación del diccionario para la posterior generación del pickle
        ITPC_tmp={ordenlabel[masc_i]:{
            "1cond":{"Umbralizacion":np.copy(Umbralizada_sig), "ClusCorr":np.copy(ClusCorr_sig)},
            "nostimvsstim":{"Umbralizacion":np.copy(Umbralizada), "ClusCorr":np.copy(ClusCorr) },
            "Values":ITPC[ordenlabel[masc_i]],
            }}
        ITPC_significance.update(ITPC_tmp)
    pkl.dump(ITPC_significance, f)
    f.close()
    return None



from time import time
monkey=33                           # Mon, solo 32 o 33
#serie=[78, 80, 84, 87, 88, 89]                           # Serie o día de registro
#orden=[3,   3, 1,   3,  3,  3]                           # Orden del set. 2 es Unc en RR032
serie=(77,)
orden=(1,)
area="S1"                           # Area de registro
computador=3

if computador==1:
    pathdir="/home/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
elif computador==2:
    pathdir="/run/media/sparra/AENHA/LFP33/FaseyPotenciaS1RR033"
else:
    pathdir="D:\\LFP32\\FaseyPotenciaS1RR032"



Parametros=(InputParameters(33, 77, 1, (2, 4, 5, 6, 7), nperm, pval), #2, 4, 5, 6, 7)),
            InputParameters(33, 78, 3, (1, 2, 4, 5, 6, 7), nperm, pval), #
            InputParameters(33, 80, 3, (1, 2, 4, 5, 6, 7), nperm, pval),     #Este
            InputParameters(33, 82, 3, (1, 2, 3, 4, 5, 6, 7), nperm, pval),  # Este
            InputParameters(33, 83, 3, (1, 2, 3, 4, 5, 6, 7), nperm, pval),  # Este
            InputParameters(33, 84, 1, (1, 2, 3, 4, 5, 6, 7), nperm, pval),  # Este
            InputParameters(33, 85, 3, (1, 2, 3, 4, 5, 6, 7), nperm, pval),  # Este
            InputParameters(33, 88, 3, (1, 2, 3, 4, 5, 6, 7), nperm, pval),  # Este
            InputParameters(33, 89, 3, (1, 2, 3, 4, 5, 6, 7), nperm, pval),#este
            InputParameters(33, 90, 3, (1, 2, 4, 5, 6, 7), nperm, pval),
            InputParameters(33, 91, 2, (1, 2, 3, 4, 5, 6, 7), nperm, pval),#este
            InputParameters(33, 92, 3, (2, 3, 4, 5, 6, 7), nperm, pval),
            InputParameters(33, 93, 3, (1, 3, 6, 7), nperm, pval),         #este
            InputParameters(33, 94, 1, (1, 2, 3, 6, 7), nperm, pval),   #Este está limpio (falta agregar),
            ) 
for i in range(len(Parametros) ):   #len(Parametros)
    for elec_i in Parametros[i].e:
        tinicial=time()
        main(Parametros[i].m, Parametros[i].s, Parametros[i].o, elec_i,  computador, Parametros[i].nperm, Parametros[i].pval)
        plt.close("all")
        tfinal=time()
        print("El tiempo de corrida es: ", tfinal-tinicial)
    