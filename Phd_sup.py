from Phd_extV2 import SLoad
from numba import njit
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap=cm.get_cmap("tab20")

def loaddataset(nmspk, nmpsyc, nmtm, lti, tf, time_return=False, orig=()):
    """
    This function loads the useful dataset based on the
    way it is stored in monkey 32 and 33.
    This function requires only the names with path to load
    such files, increse rows of data (empty lines) to match the values to the psych and time values.
    reset time to stimulus onset, add
    
    --- INPUT
    nmspk: str, namefile for spks
    nmpsyc: str, namefile for psychophysics
    nmtm: str, namefile for behavioural time.
    time_return
    orig: Es una bandera para indicar el tiempo origen a partir del cual se tomará lti y tf.
    orig como una tupla vacía en toma el tiempo donde inicia el estímulo.
    
    -----OUTPUT
    THe function returns:
    spks a list with numpy arrays containing the time spikes occurrences shifted to the third stimulus onset (main stimulus)
    psych. Psychophysics
    
    Alternatively if time_return=True, returns the behavioural times
    according to the shift in the spks.    
    """
    try:
        times=np.loadtxt(nmtm, delimiter=",")                 
        psyc=np.loadtxt(nmpsyc, delimiter=",")
    except:
        times=np.loadtxt(nmtm, delimiter="\t")
        psyc=np.loadtxt(nmpsyc, delimiter="\t")
    try: 
        A=np.loadtxt(nmspk, delimiter="\t", usecols=(0))
        delim="\t"
    except:
        A=np.loadtxt(nmspk, delimiter=",", usecols=(0))
        delim=","
    if type(orig)==tuple and len(orig)==0:
        orig=times[:, 6]
    if type(orig)==float and orig==0.0:
        orig=np.zeros(len(times), dtype=np.float32)
    A=A-1
    spks2=[]
    spks=SLoad(nmspk, delim)
    index=0
    for i in range(len(psyc)):
        if index<len(A):
            if i==A[index]:
                tmp=spks[index][1::]- orig[i]
                tmp=tmp[tmp>=lti]
                tmp=tmp[tmp<=tf]
                spks2.append(tmp)
                index+=1
            else:
                spks2.append([])
        else:
            spks2.append([])
    if time_return:
        return spks2, psyc, times-times[:, 6]
    else:
        return spks2, psyc
        
def NoiseCorr(senali, senalj, psychof, flatten=False):
    """
    This function calculates the Noise Correlation, this function
    needs the matrix of rate: senali and senalj. psychof is the classes vector
    """
    _, sizet=np.shape(senali)
    N=np.shape(senali)[0] if not flatten else np.size(senali)
    NC_t=np.zeros((sizet), dtype=np.float32)
    classes=np.unique(psychof)
    for clas_i in classes:
        if not flatten:
            pos=np.where(psychof==clas_i)[0]
            tmp_i=(senali[pos, :]-np.mean(senali[pos, :], axis=0))/(np.std(senali[pos, :], axis=0) + 1e-6 )
            tmp_j=(senalj[pos, :]-np.mean(senalj[pos, :], axis=0))/(np.std(senalj[pos, :], axis=0) + 1e-6 )
            NC_t+=np.sum(tmp_i*tmp_j, axis=0)
        else:
            pos=np.where(psychof==clas_i)[0]
            tmp_i=(senali[pos, :]-np.mean(senali[pos, :].flatten))/(np.std(senali.flatten) + 1e-6 )
            tmp_j=(senalj[pos, :]-np.mean(senalj[pos, :].flatten))/(np.std(senalj.flatten) + 1e-6 )
            NC_t+=np.sum(tmp_i*tmp_j)
    return NC_t/N
    
    
def tasaopt2(spikes, wnd, stp, ti, tf):
    """
    """
    intstep=int(wnd//stp)
    lenvec=int((tf-ti)/stp) - intstep+1
    ntrials=len(spikes)
    t=np.arange(ti+wnd, tf+stp/10, stp)
    tasas=np.zeros((ntrials, lenvec), dtype=np.float32)
    array=np.concatenate((np.arange(lenvec-1).reshape(lenvec-1, 1), np.arange(intstep, lenvec+ intstep-1).reshape(lenvec-1, 1)), axis=1 ).reshape(2*lenvec-2)
    preconteo=np.zeros(lenvec + intstep)
    for trial_i in range(ntrials):
        if len(spikes[trial_i])==0:
            continue
        preconteo[:]=0
        tmp=spikes[trial_i]-ti
        posiciones, cuentas=np.unique(np.int32(tmp//stp), return_counts=True)
        preconteo[np.int32(posiciones)]=cuentas
        tasas[trial_i, 0:-1]=np.add.reduceat(preconteo, array)[::2]
        tasas[trial_i,-1]=np.sum(preconteo[lenvec-1::])
    tasas=tasas/wnd
    return tasas, t
    
 
@njit()
def auroc(data1, basal, nbins, n1, n2, lent):
    auroct=np.zeros((lent), dtype=np.float32)
    for t_i in range(lent):
        minimo=np.min(np.array([np.min(data1[:, t_i]), np.min(basal)]))
        maximo=np.max(np.array([np.max(data1[:, t_i]), np.max(basal)]))
        binedges=np.linspace(minimo, maximo, nbins+1)
        auroc=0.0
        pxp=0       # punto en x previo (iniciales)
        pyp=0       # punto en y previo (iniciales)
        for bin_i in range(1, len(binedges)):
            px=np.sum(data1[:, t_i]<=binedges[bin_i])/n1
            py=np.sum(basal<=binedges[bin_i])/n2
            auroc+= (px-pxp)*np.min(np.array([py, pyp])) + (px-pxp)*(py-pyp)/2
            pxp=px
            pyp=py
        auroct[t_i]=auroc
    return auroct
    


def calculate_probstim(Data):
    amp, Ps=np.unique(Data, return_counts=True)
    return amp, Ps/len(Data)


def infocorr(Data, BinEdges, Amplitudes, amp):
    return None

@nb.njit()
def histograms(Data, BinEdges, Amplitudes, amp):
    PrIamp=np.zeros((len(BinEdges)-1 ,len(amp)), dtype=np.float32)
    for indice, amp_i in enumerate(amp):
        pos=np.where(Amplitudes==amp_i)[0]
        PrIamp[:, indice], _=np.histogram(Data[pos], bins=BinEdges)
        PrIamp[:, indice]=PrIamp[:, indice]/len(pos)
    return PrIamp

@nb.njit()
def Minf(PrIs, Pr, Ps, nr, nc):
    informacion=0
    for s in range(nc):
        for r in range(nr):
            if Pr[r]>0 and PrIs[r, s]>0:
                informacion+=PrIs[r,s]*Ps[s]*np.log2(PrIs[r,s]/Pr[r])
    return informacion

@nb.njit()#double [][], double[][], int, int, double[], double[], int, int )
def Minf_t(I_stim, tasas, datos, lendata, lent, Ps, amp, nr, nc):
    BinEdges=np.zeros((nr +1), dtype=np.float32)
    for t_i in range(lent):
        BinEdges[0]=np.min(tasas[:, t_i])-0.5
        BinEdges[nr]=np.max(tasas[:, t_i])+0.5
        width=(BinEdges[nr]-BinEdges[0])/nr
        for li in range(1, nr):
            BinEdges[li]=BinEdges[0]+ li*width
        Pr, _=np.histogram(tasas[:, t_i], bins=BinEdges)
        Pr=Pr/lendata
        PrIs_t=histograms(tasas[:, t_i], BinEdges, datos, amp)
        I_stim[t_i]=Minf(PrIs_t, Pr, Ps, nr, nc)  
    return None

@nb.njit()
def Minf_Perm(data, permatrix, nbins, nc, n1, n2, lent, Nperm, Ps, amps):
    infoPerm_t=np.zeros((Nperm, lent), dtype=np.float32)
    Infodata=np.ones((n1 + n2), dtype=np.int8)
    Infodata[n1::]=2
    for perm_i in range(Nperm):
        Minf_t(infoPerm_t[perm_i, :], data, Infodata[permatrix[:, perm_i]], n1+n2, lent, Ps, amps, nbins, nc)
    return infoPerm_t


@nb.njit()
def aurocPerm(data, permatrix, nbins, n1, n2, lent, Nperm):
    aurocPerm_t=np.zeros((Nperm, lent), dtype=np.float32)
    for perm_i in range(Nperm):
        data1=data[permatrix[:n1, perm_i], :]
        basal=data[permatrix[n1::, perm_i], :]
        auroct=np.zeros((lent), dtype=np.float32)
        for t_i in range(lent):
            minimo=np.min(np.array([np.min(data1[:, t_i]), np.min(basal[:, t_i])]))
            maximo=np.max(np.array([np.max(data1[:, t_i]), np.max(basal[:, t_i])]))
            binedges=np.linspace(minimo, maximo, nbins+1)
            auroc=0.0
            pxp=0       # punto en x previo (iniciales)
            pyp=0       # punto en y previo (iniciales)
            for bin_i in range(1, len(binedges)):
                px=np.sum(data1[:, t_i]<=binedges[bin_i])/n1
                py=np.sum(basal[:, t_i]<=binedges[bin_i])/n2
                auroc+= (px-pxp)*np.min(np.array([py, pyp])) + (px-pxp)*(py-pyp)/2
                pxp=px
                pyp=py
            if auroc==0:
                print(minimo, maximo)
            aurocPerm_t[perm_i, t_i]=np.abs(auroc-0.5)
    return aurocPerm_t

@nb.njit()
def entropy(dist, nelem):
    H=0
    for s in range(nelem):
        H=H -1*dist[s]*np.log2(dist[s])
    return H

def tasaopt2(spikes, wnd, stp, ti, tf):
    """
    """
    intstep=int(wnd//stp)
    lenvec=int(np.ceil((tf-ti)/stp)) - intstep+1
    ntrials=len(spikes)
    t=np.arange(ti+wnd, tf+stp/10, stp)
    tasas=np.zeros((ntrials, lenvec), dtype=np.float32)
    array=np.concatenate((np.arange(lenvec-1).reshape(lenvec-1, 1), np.arange(intstep, lenvec+ intstep-1).reshape(lenvec-1, 1)), axis=1 ).reshape(2*lenvec-2)
    preconteo=np.zeros(lenvec + intstep-1)
    for trial_i in range(ntrials):
        if len(spikes[trial_i])==0:
            tasas[trial_i]=np.nan
        else:
            preconteo[:]=0
            tmp=spikes[trial_i]-ti
            tmp=tmp[tmp>0]
            tmp=tmp[tmp<=tf-ti]
            posiciones, cuentas=np.unique(np.int32(tmp//stp), return_counts=True)
            preconteo[np.int32(posiciones)]=cuentas
            #for col_i in range(lenvec):
            #    tasas[trial_i, col_i]=sum(preconteo[col_i:col_i+intstep])
            tasas[trial_i, 0:-1]=np.add.reduceat(preconteo, array)[::2]
            tasas[trial_i,-1]=np.sum(preconteo[lenvec-1::])
            #tasas=tasas/wnd
    #tasas=tasas/wnd
    return tasas, t

@nb.njit()
def auroc(data1, basal, nbins, n1, n2, lent):
    auroct=np.zeros((lent), dtype=np.float32)
    for t_i in range(lent):
        minimo=np.min(np.array([np.min(data1[:, t_i]), np.min(basal[:, t_i])]))
        maximo=np.max(np.array([np.max(data1[:, t_i]), np.max(basal[:, t_i])]))
        binedges=np.linspace(minimo, maximo, nbins+1)
        auroc=0.0
        pxp=0       # punto en x previo (iniciales)
        pyp=0       # punto en y previo (iniciales)
        for bin_i in range(1, len(binedges)):
            px=np.sum(data1[:, t_i]<=binedges[bin_i])/n1
            py=np.sum(basal[:, t_i]<=binedges[bin_i])/n2
            auroc+= (px-pxp)*np.min(np.array([py, pyp])) + (px-pxp)*(py-pyp)/2
            pxp=px
            pyp=py
        auroct[t_i]=auroc
    return auroct

#@nb.njit()
def cluster1d(rowvec, lenvec):
    """
    Function for classifying 1d bool vectors.
    This function classifies all consecutive True elements by enumerating
    groups
    """
    classification=np.zeros((lenvec), dtype=np.int8)
    if np.max(rowvec)==0:
        return classification
    if np.sum(rowvec)>0:
        bg=1
        if rowvec[0]==1:
            classification[0]=bg        
        for index in range(1, lenvec):
            if rowvec[index]==1 and rowvec[index - 1]==1:
                classification[index]=bg
            elif rowvec[index]==1 and rowvec[index - 1]==0:
                    classification[index]=bg+1
                    bg+=1
        try:
            if np.min(classification[classification>0])==2:
                classification[classification>0]=classification[classification>0]-1
        except:
            print("Aqui andamos, ¿quś pasho?")
    return classification


@nb.njit()
def counter(row):
    """
    This function imitates the unique numpy function with the return_counts=True
    atribute. This function is essentialy used to increase the numba power and 
    accelerate code.
    Parameters
    ----------
    row : numpy 1D-array
        DESCRIPTION.

    Returns
    -------
    1d Numpy array.

    """
    unicos=np.unique(row)
    valores=np.zeros_like(unicos)
    for index, value_i in enumerate(unicos):
        if value_i!=0:
            valores[index]=np.sum(row==value_i)
    return unicos, valores


#@nb.njit()
def cluster1d_cycled(matrix, row, cols):
    """
    Function for classifying 1d bool vectors by row.
    THis function takes as basis the cluster1d function.
    """
    correction=np.zeros((row, cols), dtype=np.int8)
    for row_i in range(row):
        correction[row_i, :]=cluster1d(matrix[row_i], cols)
    return correction


def thresholding(original, permutated_sorted, Nperm, pval,  kind="absolute"):
    Npval=np.int64(Nperm*(1-pval))
    if kind=="auroc":
        threshold=np.abs(original-0.5)>=np.abs(permutated_sorted[Npval, :]-0.5)     # Thresholding original auroc
    elif kind=="absolute":
        threshold=original>=permutated_sorted[Npval, :]
    return threshold

#@nb.njit()
def significancePerm(permutated, permutated_sorted, original,  Nperm, lenoriginal, pval1=0.01, pval2=0.01, return_hist=True, return_threshold=True, kind="absoulute"):
    """
    This function corrects significative values by multiple comparisons. 
    This procedure is based on the article published by Maris & Ostenvelt, 2007. 
    
    Parameters
    ----------
    permutated : bool 2D-numpy array, every row is a permutated repetition or trial.
        DESCRIPTION.
    sig_or: bool 1D-numpy array, contains the significative bins.
        DESCRIPTION
    Nperm : TYPE
        DESCRIPTION.
    pval : TYPE, optional
        DESCRIPTION. The default is 0.01.
    thre : TYPE, optional
        DESCRIPTION. The default is (,).

    Returns
    -------
    classified, .

    """
    maximos=np.zeros((Nperm), dtype=np.int16)
    ## Epoch one: Extracting info from original calculations
    Npval=np.int16((1-pval1)*Nperm)                                             # Threshold index
    threshold=thresholding(original, permutated_sorted, Nperm, pval1, kind=kind) # Thresholding original auroc
    clus_or=cluster1d(threshold, lenoriginal)                                # Clustering threshold
    val_or, counts_or=counter(clus_or)                                          # Extracting clusters size in originals
    ## Second epoch: working with permutated values
    thr_perm=thresholding(permutated, permutated_sorted, Nperm, pval1, kind=kind)              # Thresholding permutated values
    clas=cluster1d_cycled(thr_perm, Nperm, lenoriginal)                           # Clustering permutated thresholded
    for perm_i in range(Nperm):
        unicos, valores=counter(clas[perm_i])
        maximos[perm_i]=np.max(valores)                                         # Maximum cluster per permutation
    Npval2=np.int16( (1-pval2)*Nperm)                                           # Threshold index for cluster correction.
    maximos=np.sort(maximos)
    mask=np.where(counts_or<maximos[Npval2])[0]
    if len(mask)==0:     # Everything is significative, nothing to do.
        if return_hist and return_threshold:
            return clus_or, maximos, threshold
        elif return_hist:
            return clus_or, maximos, threshold
        elif return_threshold:
            return clus_or, threshold
        else:
            return clus_or
            
    else:
        val_or=val_or[mask]    # Cut vector, mantains only smaller vectors
        for val_i in val_or:
            clus_or[clus_or==val_i]=0   # non significative groups are changed to zero.
        if return_hist and return_threshold:
            return clus_or, maximos, threshold
        elif return_hist:
            return clus_or, maximos, threshold
        elif return_threshold:
            return clus_or, threshold
        else:
            return clus_or    

