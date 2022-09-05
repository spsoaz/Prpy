# Modulo escrito por:
# Sergio Parra sanchez

# Instituto de Fisiologi­a Celular, Universidad Nacional Autonoma de Mexico

# Carga libererias
import numpy as np
import numba as nb
import scipy.fft as fft
import scipy.signal as sn
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
#from scipy import stats
#import statsmodels.api as sm
#import numba as nb

#direction="/home/sparra/AENHA/LFP33/LFP_RR033/S1_izq_MAT/RR033077_001"

# Load lfp data
def loadlfp(direction):
    """
    This function loads mat files with the information of lfp. COnsider that this functions could function
    only for data of monkey 32 and 33 due to the conversion made for the functions for the sorting, 
    so probably would be necessary to do another function for lfp loading when
    we analyze data of other ares of the same monkey or even data from other monkeys.
    This function requires the input direction:
    direction is a string with the path to the files of the session.
    example:
    data=loadlfp("/home/sparra/AENHA/LFP33/LFP_RR033/S1_izq_MAT/RR033075_001")
    """
    from scipy.io import loadmat
    import os
    files=os.listdir(direction)  #list with the file names or directorynames as elements
    files.sort()                 #Sort filenames, then data will be loaded in order.
    trial=[]
    for i in range(len(files)):
        if files[i].find("Analogdatafile")!=-1:
            data=loadmat(os.path.join(direction, files[i]))
            data=data["Analog"]
            lfp=np.zeros((data[0][0][-2][0][0], 7), dtype=np.float32)
            for elec in range(7):
                lfp[:, elec]=data[elec][0][-1][:, 0]
            trial.append(lfp)
    return trial
    
def nsx2txt(datafile, name, savef=False, **kwargs):
    """
    Function writen by the blackrock microsystems group and lightely adapted
    by Sergio Parra Sanchez.
    
    This function loads and optionaly writes a file with the data contained
    in a nsx file.
    
    """

    from numpy               import arange
    from brpylib             import NsxFile, brpylib_ver

# Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

# Inits
 #   datafile = '/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Nev/Nev/RR032152_006/datafile1026.ns3'
    elec_ids     = 'all'  # 'all' is default for all (1-indexed)
    start_time_s = 0                       # 0 is default for all
    data_time_s  = 'all'                      # 'all' is default for all
    downsample   = 1                       # 1 is default
    plot_chan    = 6                       # 1-indexed

# Open file and extract headers
    nsx_file = NsxFile(datafile)

# Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
    cont_data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)

# Close the nsx file now that all data is out
    nsx_file.close()

# Plot the data channel
    ch_idx  = cont_data['elec_ids'].index(plot_chan)
#hdr_idx = cont_data['ExtendedHeaderIndices'][ch_idx]
    t       = cont_data['start_time_s'] + arange(cont_data['data'].shape[1]) / cont_data['samp_per_s']

#    plt.plot(t, cont_data['data'][ch_idx])
#    plt.axis([t[0], t[-1], min(cont_data['data'][ch_idx]), max(cont_data['data'][ch_idx])])
#    plt.locator_params(axis='y', nbins=20)
#    plt.xlabel('Time (s)')
#    plt.ylabel('Output');
#    plt.title('Electrode 6');
#    plt.show()

#This section writes data into in a txt file where the first column is time and the following are every one of
#the channels of recording
    if savef:
        data_txt=open(name,"w");
        for j in range(0,cont_data['data'].shape[1]):
            data_txt.write("%f\t" % t[j])
            for i in range(0,cont_data['data'].shape[0]):
                data_txt.write("%f\t" % cont_data['data'][i][j])
                if i==cont_data['data'].shape[0]-1:
                    data_txt.write("\n")
        data_txt.close
    return cont_data, t



def loadnsx(direction):
    """
    This function loads mat files with the information of lfp. 
    This function requires the input direction:
    direction is a string with the path to the files of the session.
    example:
    data=loadlfp("/home/sparra/AENHA/LFP32/A1_izq_MAT/RR032189_002")
    """
    import os
    files=os.listdir(direction)  #list with the file names or directorynames as elements
    files.sort()                 #Sort filenames, then data will be loaded in order.
    trial=[]
    for i in range(len(files)):
        if files[i].find("ns3")!=-1:
            data, t=nsx2txt(os.path.join(direction, files[i]), " ")
            data=data["data"]
            lfp=np.zeros((np.size(data, axis=1), 7), dtype=np.float32)
            for elec in range(7):
                lfp[:, elec]=data[elec, :]
            trial.append(lfp)
    return trial, t                
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#                                       Metodos basados en la transformada de Fourier
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# FFT by the multitaper method using slepian tapers
def fftmultitaper(data, srate, bw):
    '''
      Metodo para el calculo del espectro de una senhal basado en el metodo de los multiTapers de 
      Thomson. Esta funcionn hace uso de la secuencia slepiana calculada en el mondulo de scipy.signal.windows.
     
      Esta funcion requiere de las siguientes entradas:
      data: Serie de tiempo, un arreglo unidimensional de numpy
      srate: Frecuencia de muestreo
      bw:    Ancho de banda para estimar el numero de tapers slepianos ntapers=bw*2-1
     
      Esta funcion devuelve el espectro de potencia escalado al numero de datos.
    '''
    from scipy.signal.windows import dpss
    # define Slepian tapers.
    n=len(data)
    ntap=bw*2-1
    tapers = dpss(n, bw, ntap)
    # initialize multitaper power matrix
    mtPow = np.zeros((n//2+1))
    hz = np.linspace(0,srate/2, n//2+1);

    # loop through tapers
    for tapi in range (np.size(tapers,axis=0)-1):# % -1 because the last taper is typically not used

        # scale the taper for interpretable FFT result
        temptaper = tapers[tapi,:]/np.max(tapers[tapi,:]);

        # FFT of tapered data
        x = abs(2*fft.fft(data*temptaper)/n)**2;

        # add this spectral estimate to the total power estimate
        mtPow[:]+= x[0:len(hz)]
    # Because the power spectra were summed over many tapers,
    # divide by the number of tapers to get the average.
    return mtPow[:]/ntap
# Funcionn para calcular el espectrograma mediante la transformada corta de FOurier

def Spectrogram(data, srate, width, noverlap=0.2, plotting=False, window=(), **kwargs):
    '''
        Esta funcionn realiza un espectrograma mediante la ticnica de la transformada de Fourier,
        la funcionn utiliza el taper de Hamming exclusivamente hasta el momento
        data: numpy array
        srate: Frecuencia de muestreo, escalar
        width: Ancho de la ventana en segundos
        noverlap: Traslape en fraccionn [0, 1], default: 0.2 (20%)
        plotting: bool, True muestra el grafico, default:False
          -- cuando plotting=True
            la salida es:
              1. Frecuencias
              2. Tiempo 
              3. TFmatrix: Matriz de tiempo-frecuencia
          -- cuando plotting=False
            la salida es:
              1. Frecuencias
              2. Tiempo 
              3. TFmatrix: Matriz de tiempo-frecuencia
              4. im, Una lista con los elementos de la figura y el arreglo de ejes.
        window: Este parametro es para indicar el tipo de ventana que se utilizara en el proceso
                de la creacionn del mapa de tiempo frecuencia.Se trata de un vector con la misma longitud 
                que el tamanho de la ventana para ir utilizandolo a modo de taper.
                Si la longitud es cero(default) se utilizara una ventana de Hamming. 
                Una tercera opcionn es que la ventana sea de tipo tuple de dos elementos, 
                el primer elemento sera de tipo str e igual a mtm 
                mientras que el segundo debera ser un escalar de tipo entero indicando el ancho
                de banda para calcular los tapers Ntapes=2*bw+1 
                donde bw es el segundo elemento
                
       
        **Ejemplo:
        srate=1000
        t=np.arange(0, 5, 1/srate)
        n=len(t)
        f=(30, 3, 6, 12)
        tchunks=np.int32(np.round(np.linspace(0, n, len(f)+1)))
        #print(tchunks)
        data=np.zeros((0))
        for i in range(len(f)-1):
            data=np.concatenate((data, np.sin(2*np.pi*f[i]*t[tchunks[i]:tchunks[i+1] ])), axis=0)
        data=np.concatenate((data, np.sin(2*np.pi*f[-1]*t[tchunks[-2]:: ])), axis=0)
        
        hz, t, TFmatrix, im=Spectrogram(data, srate, width, noverlap=0.7, plotting=True)
    '''

    width_n= np.int32(width*srate/2)   #Ancho en ti©rminos del numero de puntos dada la frecuencia de muestreo7
    paso=np.int32(np.round(width*srate*(1-noverlap)))
    hz=np.linspace(0, srate/2, width_n+1)
    centimes=np.int32(np.round(np.arange(width_n, len(data)-width_n, paso)))        
    if type(window)==tuple and   len(window)==0:   #Mi©todo default
        hammingWin=np.hamming(width_n*2)
    else:
        hammingWin=window
    TFmatrix=np.zeros((len(centimes), len(hz)), dtype=np.float32)
    if type(window)==tuple and len(window)==2:
        if type(window[0])==str and type(window[1])==int:   #Uso del mi©todo multitaper
            for tf in range(len(centimes)-1):
                Xdata=data[centimes[tf]-width_n: centimes[tf] + width_n ]
                Xdata=fftmultitaper(Xdata, srate, bw=window[1])
                TFmatrix[tf, :]=Xdata
        else:
            raise ValueError("Vea documentacionn, window no esta bien definido")

    elif type(window)==tuple and   len(window)==0:
            for tf in range(len(centimes)-1):
                Xdata=hammingWin*data[centimes[tf]-width_n: centimes[tf] + width_n ]
                Xdata=fft.rfft(Xdata)
                Xdata=2*np.abs(Xdata[0:len(hz)])/width_n
                TFmatrix[tf, :]=Xdata
    if not plotting:
        return hz, centimes/srate, np.transpose(TFmatrix)
    else:
        #Plotting
        fig, ax=plt.subplots(2, 1, figsize=(12, 9))
        ax[0].plot(np.arange(0, len(data)/srate, 1/srate), data)

        #cbar=ax[1].contour(t[centimes], hz, TFmatrix.transpose(), levels=1000, cmap="viridis")
        #ax[1].set_ylim([0, 40])
        #ax[1].set_xlim([0, 5])
        #plt.colorbar(cbar)
        cbar=ax[1].imshow(TFmatrix.transpose(), aspect="auto", cmap='hot', extent=[centimes[0]/srate, centimes[-1]/srate, hz[0],hz[-1]], interpolation='gaussian', origin="lower")
        ax[1].set_ylim([0, 120])
        ax[1].set_xlim([0, 5])

        plt.colorbar(cbar)

        plt.tight_layout()
        ax[1].set_xlabel("Tiempo [s]", fontsize=18)
        ax[1].set_ylabel("Frecuencias [Hz]", fontsize=18)
        ax[0].tick_params(labelsize=18)
        ax[1].tick_params(labelsize=18)
        return hz, centimes/srate, np.transpose(TFmatrix), [fig, ax]

# #Function for spectrogram with fft with variable width.
# #Power spectra (5â€“40 Hz) were computed using afast Fourier transform approach. 
#Trials were segmented into0.5-s epochs of the relevant events, padded 
#with zeros to 1-slength and multiplied with a Hanning taper, to improve thespectral estimation.
# Furthermore, to inspect the time course ofthe LFP oscillations, we computed time
#“frequency representa-tions of power using an adaptive sliding time window of 0.5-slength
#multiplied with a Hanning taper.



def AdapWinSpec(senal, fsampling, noverlap, Ncicles=5,  freqs="linear",  window=( ), **kwargs):
    """
    This function performs a spectrogram through short Fourier Transform by using 
    windows lengths as functions of the frequency. This function is expensive due to every time-frequency 
    bin is calculated element by element.    

    Parameters
    ----------
    senal : 1- numpy array
        Es un array de numpy unidimensional. If the input is a matrix Fourier transform is calculated along
        the axis=1.
    fsampling : int
        Frequency sampling in Hz.
    noverlap : float
        Overlap proportion between successive time windows. This overlap applies only for the 
        largest time-winds (low frequency)
    Ncicles : int, optional
        The length of the window in cycles of the lowest frequency. The default is 5.
    freqs : 1-d array or str (linear or log), optional
        If is a string parameter, the freqs are built as a linspace from 4 to nyquist frequency. The default is "linear".
        When the parameter is log, it is the same but with a logaritmic space between succesive frequencies.
        If is 1-d array. These are the different frequencies to take into account. Program does not verifies that
        array is according to the shanon theorem.
    window : str or tuple, optional
        Indicates the type of window to use as a taper.Uses the same winwdows names as the
        Scipy function, The default is ( ), indicating a Hamming window.
        To include the mutitaper method proposed by thompson it is necessary to
        use a tuple where the elements are 0: a string, 1 the bandwidth of the multitaper method.
        In this configuration, function uses the function multitaper. Notice that
        uses this method for all frequencies, not for the greater ones only.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # Aun queda verificar que la senal sea unidimensional, este programa solo admite una senal a la vez.
    nyquistfreq=fsampling/2                  # Frecuencia de Nyquist
    if type(freqs)==str:
        if freqs=="linear": 
                hz=np.linspace(4, nyquistfreq, (len(senal)//2 +1)//4)    # Generacion de frecuencias espaciadas linealmente
        elif freqs=="log":  
                hz=np.logspace(np.log10(4), np.log10(nyquistfreq), (len(senal)//2 +1)//8)  # Generacion de frecuencias espaciadas logari­tmicamente
        else:
            raise ValueError("Las frecuencias solo se mantienen como lineales o logaritmicas, vea documentacion.")
    elif (type(freqs)==list or type(freqs)==tuple or type(freqs)==type(np.zeros((0))) and np.size(freqs, axis=0)>1):
            hz=freqs
    else:
        raise ValueError("Las frecuencias no son del tipo o estructura necesaria. Vea la documentacion")
    maxwidth=np.int32(Ncicles/hz[0]*fsampling//2 )       # Semiancho maximo 
    if type(window)==tuple and   len(window)==0:         # Metodo default
        window=np.hamming(maxwidth*2)
    else:
        None
    cent=np.arange(maxwidth, len(senal)- maxwidth, np.int32((1-noverlap)*maxwidth*2))
    TFmatrix=np.zeros((len(hz), len(cent)), dtype=np.float32)
    if type(window[0])==str and type(window)==tuple and type(window[1])==int:   #  Uso del metodo multitaper
        for frq in range(len(hz)):    
            width=np.int32(Ncicles/hz[frq]*fsampling//2)
            freqs=np.linspace(0, nyquistfreq, width+1)
            pos=np.argmin(np.abs(freqs-hz[frq]))               
            for ti in range(len(cent)) :
                senalX=senal[cent[ti]-width:cent[ti] + width]
                senalX=fftmultitaper(senal[cent[ti]-width:cent[ti] + width], fsampling, bw=window[1])
                TFmatrix[frq, ti]=senalX[pos]                                      
        return cent/fsampling, hz, TFmatrix
    
    else:
    # Uso de solo un taper metodo del periodograma
        for frq in range(len(hz)):    
            width=np.int32(Ncicles/hz[frq]*fsampling//2)
            freqs=np.linspace(0, nyquistfreq, width+1)
            pos=np.argmin(np.abs(freqs-hz[frq])) 
            if(np.abs(freqs[pos] - hz[frq])/hz[frq])>0.15:
                # None
                print("Error en la estimacion de la frecuencia, los valores tienen un error mayor al 15%. Considere incrementar el ancho de la ventana", np.abs(freqs[pos] - hz[frq])/hz[frq])
            window=np.hamming(2*width)
            for ti in range(len(cent)) :
                senalX=np.abs(2*fft.rfft((senal[cent[ti]-width:cent[ti] + width])*window)/(2*width))**2
                if len(senalX)!=len(freqs):
                    print(len(senalX), len(freqs))
                    raise ValueError("No coinciden las frecuencias estimadas y calculadas")
                TFmatrix[frq, ti]=senalX[pos]                                      
        return cent/fsampling, hz, TFmatrix


# """
# Funcion realizada por Sergio Parra Sanchez, copyright.
# Funcion para normalizar un espectrograma o un conjunto de datos. 
# Esta funcionn recibe una matriz de datos (espectrograma) o un espectro de potencia y lo normaliza 
# segun tres opciones: transformada z, conversion a decibeels o cambio porcentual.
# La funcionn recibe los siguientes parametros:

# * data: Un ndarray conteniendo una arreglo bidimensional (espectrograma) o bien un espectro de potencia.
# * srate: Frecuencia de muestreo
# * segment: es una tupla que indicara cual es el baseline para realizar la normalizacion.
#   - Por default es una tupla vaci­a que considera los primeros 300 ms como baseline.
#   - Si se introduce un array con dos elementos estos seran los tiempos de inicio y final, en ese orden, 
#   que delimitaran el segmento para realizar la normalizacion.
#   - Si se introduce un np.ndarray o un array de otra tamano distinto a dos considerara que
#   ese es el segmento para calcular el baseline. El baseline es un array si data es un array o una matriz
#   si data es una matriz. Arreglos de mas de dos dimensiones no se consideran.
# * whtype: Indica el tipo de normalizacion:
#     - "db", normalizacion por conversion a decibeles
#     - "trz" normalizacion con transformada Z
#     - "percnt" normalizacion por cambio porcentual.    
# """


def normalize(data, srate, segment=(), whtype="db", **kwargs):
    """
    Normalization function for a scalogram or spectrogram.
    This function receives a matrix data (spectrum as time function). Normalization
    is done according to three different types.

    Parameters
    ----------
    data : 2-d array 
        Data matrix,  Rows ar frequencies and columns time FXT.
    srate : int
        DESCRIPTION.
    segment : tuple, optional
        Indicates the baseline for normalization. The default is ().
    whtype : str, optional
        Indicates the type of normalization, there are three options:. The default is "db".
        * "db" decibel conversion
        * "trz" Z transform normalization
        * "percnt" Percentual change 
    **kwargs : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    2-d array
        Normalized data.

    """

    if (type(data)==list):    # data es una lista
        if type(data[0])==list:
            data=np.ndarray(data)
        else:
            raise TypeError("Data type not supported, see documentation.") 
    elif type(data)==tuple:   #data es una tupla
        raise TypeError("Data type not supported, data must be a 2d-numpy.ndarray")
    elif type(data)==type(np.zeros((0))) and np.size(data, axis=1)>1 and np.size(data, axis=0)>1 and len(data.shape)==2:  # Numpy array  
        None  # good
    else:
        raise TypeError("Data type not supported, see documentation.")
    
    # From now on, data must be a 2 ndarray 
    
    # Now extract the average baseline for normalization purposes.
    if type(segment)==tuple and len(segment)==0:   #Default parameter
        tf=np.int32(0.2*srate)
        ti=0
        baselineav=np.mean(data[:, ti:tf], axis=1)
    elif len(segment)==2 and (type(segment)==list or type(segment)==tuple) and isinstance(segment[0], int) and isinstance(segment[0], float):                            # segment is list or tuple with only two elements
        if segment[1]>segment[0]:
            ti=np.int32(segment[0]*srate)
            tf=np.int32(segment[1]*srate)
        else:
            raise ValueError("segment[1] must be bigger than segment[0]")
        baselineav=np.mean(data[:, ti:tf], axis=1)  #
    elif type(segment)==type(np.zeros((0))) and len(np.shape(segment))==2:  # segment is a 2d numpy.ndarray
        if np.size(segment, axis=0) == np.size(data, axis=0):
            baselineav=np.mean(segment, axis=1)
        else:
            print("Rows of data and segment are not equal, possibly error")
            baselineav=np.mean(segment, axis=1)
    else:
        raise TypeError("segment is not a data type supported, see documentation")
        
    # Now considers the type of normalization   
    baselineav=np.repeat(np.reshape(baselineav, (len(baselineav), 1) ), np.size(data, axis=1), axis=1)
    if whtype=="db":
        return 10*np.log10(data/baselineav)
    elif whtype=="trz":   
        baselinestd=np.std(data[:, ti:tf], axis=1)
        baselinestd=np.repeat(np.reshape(baselinestd, (len(baselineav), 1) ), np.size(data, axis=1), axis=1)
        return (data-baselineav)/baselinestd
    elif whtype=="percnt":
        return 100*((data-baselineav)/baselineav)
    else:
        raise ValueError("Wrong type parameter, see documentation.")

# # #----------------------------------------------------------------------------------------------------
# # #----------------------------------------------------------------------------------------------------
# # #         Herramientas de preprocesamiento Test de estacionaridad y limpieza de ruido
# # #----------------------------------------------------------------------------------------------------
# # #----------------------------------------------------------------------------------------------------

#@nb.njit()
def estamean(senal, fsamp, window, return_times=False, **kwargs):
    """
    Esta funcionn devuelve la media por ventanas no traslapadas con fines de
    verificacionn de estacionariedad en la media. ademas
    devuelve los valores de tiempo de acuerdo a la senal. La funcionn requiere 
    los siguientes argumentos de entrada:
    * senal: La senhal, un numpy array
    * fsamp: Frecuencia de muestreo
    * window: Tamanho de la ventana en segundos
    """
    wn=np.int32(window*fsamp//2)
    centimes=np.arange(wn, len(senal)-wn, wn)
    statmean=np.zeros((len(centimes)), dtype=np.float32)
    for i in range(len(centimes)):
        statmean[i]=np.mean(senal[centimes[i]-wn:centimes[i]+wn])
    if return_times:    
        return centimes/fsamp, statmean
    else:
        return statmean
    
#@nb.njit()
def estavar(senal, fsamp, window, return_times=False, **kwargs):  
    """
    Esta funcionn devuelve la varianza por ventanas no traslapadas con fines de verificacionn de estacionariedad en la media. ademas
    devuelve los valores de tiempo de acuerdo a la senal. La funcionn requiere los siguientes argumentos de entrada:
    * senal: La senhal, un numpy array
    * fsamp: Frecuencia de muestreo
    * window: Tamanho de la ventana en segundos
    """
    wn=np.int32(window*fsamp//2)
    centimes=np.arange(wn, len(senal)-wn, wn)
    statvar=np.zeros((len(centimes)), dtype=np.float32)
    for i in range(len(centimes)):
        statvar[i]=np.var(senal[centimes[i]-wn:centimes[i]+wn])
    if return_times:    
        return centimes/fsamp, statvar
    else:
        return statvar

def estmultcov(senales, fsampm, window, return_times=False, metric="euclidean", **kwargs):
    """
    Esta funcionn devuelve un analisis de estacionaridad por covarianzas en una senhal multicanal. 
    Los parametros de entrada son:
    * senales
    * fsamp
    * window
    * return_times
    * metric    
    """
    print("Upsss en contruccionn")
    return 0





# """
#   EMD - Realiza una descomposicionn empi­rica de modos de una serie de datos o senhal.

#   **Uso:**
#   imfs = emd(data , maxorder=30, maxstd=0.5, maxite=1000)

# **Argumentos de entrada:**
#     data     = Serie de datos (un numpy array)
#     maxorder = (opcional) Orden maximo del EMD (default es 30)
#     maxstd = (opcional) Desviacionn estandar del criterio de sifting (default 0.5)
#     maxiter  = (opcional) Numero maximo de iteraciones, valor por default 1000

#   **Salida :**
#     imfs    = [modos X tiempo] Matriz con los modos intri­nsecos

# """

def emd(data, maxorder=30, maxstd=0.5, maxiter=1000, **kwargs):

    import sys
    eps=sys.float_info.epsilon
    npnts = len(data)
    t  = np.arange(0, npnts)
    # Inicializacionn
    imfs  = np.zeros((maxorder, npnts), dtype=np.float32);

    ## use griddedInterpolant if exists (much faster than interp1)

    # griddedInterpolat should always be preferred when your Matlab version supports it.

    # loop over IMF order

    imforder = 0
    stop     = False
    imfsignal=np.copy(data)
    while not stop:

        ## Iteracionn sobre el proceso de sifting

        # Inicializacion
        standdev = 10
        numiter  = 0
        signal   = np.copy(imfsignal)

        # "Sifting" means iteratively identifying peaks/troughs in the
        # signal, interpolating across peaks/troughs, and then recomputing
        # peaks/troughs in the interpolated signal. Sifting ends when
        # variance between interpolated signal and previous sifting
        # iteration is minimized.
        while standdev>maxstd and numiter<maxiter:
            # Identificacionn de mi­nimos y maximos locales
            localmin  = np.concatenate(([0], np.where(np.diff(np.sign(np.diff(signal)))>0)[0]+1))
            localmin  = np.concatenate((localmin, [npnts-1]))
            localmax  = np.concatenate(([0], np.where(np.diff(np.sign(np.diff(signal)))<0)[0]+1))
            localmax  = np.concatenate((localmax, [npnts-1]))
          
            FL = griddata(t[localmin], signal[localmin], xi=t, method='cubic');
            FU = griddata(t[localmax], signal[localmax], xi=t, method='cubic');

            # compute residual and standard deviation
            prevsig   = signal;
            signal    = signal - (FL + FU)/2;
            standdev  = np.sum( ((prevsig - signal)**2) / (prevsig**2 + eps) ); # eps prevents NaN's

            # not too many iterations
            numiter = numiter+1;

        # end sifting

        # imf is residual of signal and min/max average (already redefined as signal)
        imfs[imforder,:] = signal;
        imforder = imforder+1;

        ## residual is new signal

        imfsignal = imfsignal-signal;

        ## stop when few points are left
        #print("El numero de maximo locales es: ", len(localmax))
        #print("El numero de mi­nimos locales es: ", len(localmin))
        localcirit=np.min([len(localmax), len(localmin)])
        if localcirit<=5 or imforder>=maxorder:
            stop=True;
       
    return imfs

def g(x):
    return np.tanh(x)


def g_der(x):
    return 1 - g(x) * g(x)

def whitening(x):
    """
    Funcion que permite realizar parte del preprocesamiento de la matriz
    de senales.
    """
    cov = np.cov(x)

    d, E = np.linalg.eigh(cov)

    D = np.diag(d)

    D_inv = np.sqrt(np.linalg.inv(D))

    x_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, x)))

    return x_whiten


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w

    w_new /= np.sqrt((w_new ** 2).sum())

    return w_new

def center(x):
    x = np.array(x)
    
    mean = x.mean(axis=1, keepdims=True)
    
    return x - mean

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


# """
#     SSA - Realiza una descomposicionn mediante la ti©cnica de Spectral Singular Analisis.

#   **Uso:**
#   yrec, y_comp, D = ssa(data , window=0.5, sv_ratio=0.5, return_eig=True)
  
#   yrec, y_comp = ssa(data , window=0.5, sv_ratio=0.5, return_eig=False)

# **Argumentos de entrada:**
#     data     = Serie de datos (un numpy array)
#     srate    = Frecuencia de muestreo
#     window   = (opcional) Tamanho en segundos de la ventana, default 0.5 seg.
#     sv_ratio = (opcional) Calidad del filtrado, entre mas cerca de 1 mas filtrado, default 0.5
#     return_eig  = (opcional) Devuelve los eigenvalores para estimar el numero necesario, default=False

#   **Salida :**
#     y_rec    = [senhal] Arreglo de numpy con la reconstruccionn de la senhal
#     y_comp   = [Componentes, time] Matriz que devuelve los componentes principales utilizados en la reconstruccionn
#     D        = [serie]  Arreglo de numpy con los eigenvalores de la matriz Hankeliana producida en el embedding.

# """


def ssa(snl1, srate, window=0.5, sv_ratio=0.5, return_eig=False, **kwargs):

# Paso 1: Embedding
    # El Embedding para generar la matriz de Hankel
    N=len(snl1)
    k=np.int64(window*srate)
    j=np.int64( N + 1 - k)
    Hankel=np.zeros((k, j), dtype=np.float32)
    for i in range(k):
        Hankel[i, : ] = snl1[i: i + j]

    # 1.2 Descomposicionn en valores singulares
    U, D, VT=np.linalg.svd(Hankel, full_matrices=True)
    #Dcumsum=np.cumsum(D)
    ## Paso 2: Reconstruccionn:
    n=np.max(np.where(np.diag(D)/D[0]>sv_ratio)[0])  # Seleccionn de las primeras componentes
    #print("Se tomaran los %d componentes"%(n))
    Y =np.dot(U[:,0: n ], np.diag(D[0:n]))#*VT[: , 0: n ];
    Y=np.dot(Y, np.transpose(VT[:, 0:n]))  # Primer grupo 
    # Hankelizacionn mediante la suma sobre las diagonales de Hankel (i + j=const)
    yrec=np.zeros((N), dtype=np.float32)
    yrec[0] =Y[0, 0]
    suma=1
    for i in range(1, np.size(Y, axis=1)): #Hacemos todas las columnas 
        min_il =np.min([i, np.size(Y, axis=0)-1]) #l=4,  k=un resto, i=5
        ren=np.arange(min_il, -1,-1, dtype=np.int64)
        yrec[i]=np.mean(Y[ren, np.int64(suma-ren)])    
        suma+=1
    # Aqui­ haremos las ultimas columnas
    ending=0
    for i in range(np.size(Y, axis=0)-2): # La ultimas antidiagonales
        ren=np.arange(np.size(Y, axis=0)-1, ending , -1)
        yrec[ np.size(Y, axis=1) +i]=np.mean(Y[ren,  np.int64(suma-ren)])
        suma+=1
        ending+=1
    yrec[-1]=Y[-1, -1]  
    # Exraccionn de los componentes principales de la senhal
    y_comp=np.zeros((n, N), dtype=np.float32)
    for kk in range(n):
        Y2=U[:, kk]*D[kk]
        #print("U  es de tamao; ", U.shape )
        #print("VT  es de tamao; ", VT.shape )
        Y2=np.dot(np.reshape(Y2, (k, 1)), np.reshape(VT[:, kk], (1, j)))  # Primer grupo 
        #print("y2 es de tamao ", Y2.shape)
        y_comp[kk, 0]=Y2[0, 0]
        suma=1
        for i in range(1, np.size(Y2, axis=1)): #Hacemos todas las columnas 
            min_il =np.min([i, np.size(Y2, axis=0)-1]) #l=4,  k=un resto, i=5
            ren=np.arange(min_il, -1,-1, dtype=np.int64)
            y_comp[kk, i]=np.mean(Y2[ren, np.int64(suma-ren)])    
            suma+=1
        # Aqui­ haremos las ultimas columnas
        ending=0
        for i in range(np.size(Y2, axis=0)-2): # La ultimas antidiagonales
            ren=np.arange(np.size(Y2, axis=0)-1, ending , -1)
            y_comp[kk, np.size(Y2, axis=1)+i]=np.mean(Y2[ren,  np.int64(suma-ren)])
            suma+=1
            ending+=1
        y_comp[kk, -1]=Y2[-1, -1] 
    if return_eig:
        return yrec, y_comp, D
    else: 
        return yrec, y_comp

# """
# Esta funcionn realiza el rerefernciado offline a un conjunto de senhales registradas 
# simultaneamente. La funcionn rereferencia mediante el mi©todo de common average.

# La funcion requiere de los siguientes parametros de entrada:

# 1. **matrixsignals**: Es una numpy array donde las columnas son los electrodos mientras que
# los renglones es el tiempo de registro. TXN

# """

def rreferencing(matrixsignals):

    time, nelec=np.size(matrixsignals)
    return matrixsignals-np.mean(matrixsignals, axis=1)
    
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
#                                                   Metodos basados en wavelets
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------





# """
# Esta funcion realizara una wavelet de morlet.
# La wavelet de Morlet se define de la siguiente manera:

# $mwv=\frac{1}{\sqrt{s\sqrt{\pi}}} e^{\frac{-t^2}{2s^2}}e^{i2\pi ft}$

# Donde:
# $s=\frac{\#Ciclos}{2\pi f}$.

# La función requiere de los siguientes parámetros:
# fsamp: Frecuencia de muestreo
# Nciclos: Numero de ciclos
# f: Frecuencia en la cual se centrara.
# t: Opcional, duracion temporal, esta se mantiene en dos segundos por default [-2, 2].
# fwhm: Opcional, el ancho espectral como filtro para la senhal. Por default es False.
# rsp: Opcional, devuelve el espectro de potencia de la wavelet compleja. Default=False.

# [1] Cohen Mike X. Analyzing Neural Time Series Data, Theory and Practice, MIT Press 2014. Chap 13


# Como ejemplos tiene las siguientes funciones:

# ** Ejemplo 1:
#     wavelet, fwhm = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=True, rsp=False)
# ** Ejemplo 2:
#     wavelet = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=False, rsp=False)
# ** Ejemplo 3:
#     wavelet, fwhm, mwaveletX = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=True, rsp=True)
# ** Ejemplo 4:
#     mwavelet, mwaveletX = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=False, rsp=False)        
# """

def WaveletMorlet(fsamp, Nciclos, f, t=2, fwhm=False, rsp=False, **kwargs):

    if t<=0:
        raise ValueError("La duracion debe ser positiva, vea la documentacion.")
    if Nciclos<=0:
        raise ValueError("Los ciclos deben ser estrictamente positivos, vea la documentacionn")
    t=np.arange(-t, t, 1/fsamp)
    w=2*(Nciclos/(2*np.pi*f))**2
    mwavelet=np.exp(1j*2*np.pi*f*t) * np.exp((-t**2)/w) #onda compleja * Gaussiana
    if not (np.abs(mwavelet[0])<=0.01 and np.abs(mwavelet[-1])<=0.01):
        raise ValueError("La duracionn de la wavelet no es suficiente para amortiguar la senhal, considere incrementar la duracionn")
    
    if fwhm: #Devuelve el fwhm
        mwaveletX=fft.fft(mwavelet)
        mwaveletX=mwaveletX*np.conj(mwaveletX)
        mwaveletX/=np.max(mwaveletX)
        freqs=np.linspace(0, fsamp/2, len(mwavelet)//2+1)  #Calcula las frecuencias
        fwhmM=np.argmax(mwaveletX)  #Encuentra la posicion del maximo
        fwhm_vi=np.argmin(np.abs(mwaveletX[0:fwhmM]-0.5))
        fwhm_vd=np.argmin(np.abs(mwaveletX[fwhmM::]-0.5))
        if rsp:
            return mwavelet, mwaveletX, freqs[fwhm_vd+fwhmM]-freqs[fwhm_vi]            
        else:
            return mwavelet, freqs[fwhm_vd+fwhmM]-freqs[fwhm_vi]
    else: 
        if rsp:
            mwaveletX=fft.fft(mwavelet)
            mwaveletX=mwaveletX*np.conj(mwaveletX)
            mwaveletX/=np.max(mwaveletX)
            return mwavelet, mwaveletX
        else:
            return mwavelet
    

    
def MorletWaveletSpectrogram(senal, fsampling, Ncicles=5, freqs="linear", lentimewav=3, **kwargs):
    # """
    # Esta funcion realiza el espectrograma de una sñal mediante la convolucion con wavelets de Morlet complejas,
    # la funcionn recibe los siguientes parametros de entrada:
    # * senal:    Es un array de numpy unidimensional
    # * fsampling Frecuencia de muestreo en Hz
    # * Ncicles   Numero de ciclos, si es un entero, todas las frecuencias utilizan el mismo numero de ciclos. Si es un array, entonces debera ser del mismo tamanho que las frecuencias freqs
    # * freqs     frecuencias, si es una cadena de caracteres, solo puede tomar los siguientes valores:
    #       - "linear", genera una serie de frecuencias con equiespaciamiento. Si Ncicles es un entero, entonces utilizara el eauivalente a len(senal)//4 + 1 frecuencias que van de 1 a Nyquistfreq-100 .
         
    #       - "log" De manera similar a "linear" pero con un espaciamiento logari­tmico.
         
    #       - array: Si es un array, entonces indicara las frecuencias que se analziaran y ademas debe ser 
    #       igual al numero de elementos en Ncicles si este no es un entero, si es un entero entonces todas 
    #       las frecuencias llevaran el mismo numero de ciclos.
    # * lentimewav: Es la semiduracionn de la ondi­cula, por default es de 3 segundos.
    
    # Esta funcion no verifica que la senhal sea unidimensional (por el momento), de modo que es requisito del usuario asegurar este hecho hasta el momento.
    # """
    
    # Aun queda verificar que la senal sea unidimensional, este programa solo admite una senhal a la vez.
    nyquistfreq=fsampling/2                  # Frecuencia de Nyquist
    if type(freqs)==str:
        if freqs=="linear": 
            if type(Ncicles)==int:
                hz=np.linspace(1, nyquistfreq-100, len(senal)//4 +1)    # Generacionn de frecuencias espaciadas linealmente
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
            elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
                hz=np.linspace(1, nyquistfreq-100, len(Ncicles))
        elif freqs=="log":  
            if type(Ncicles)==int:
                hz=np.logspace(np.log10(1), np.log10(nyquistfreq-100), len(senal)//4 +1)  # Generacionn de frecuencias espaciadas logari­tmicamente
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
            elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
                hz=np.logspace(np.log10(1), np.log10(nyquistfreq-100), len(Ncicles))
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
        elif   (type(freqs)==list or type(freqs)==tuple or type(freqs)==type(np.zeros((0))) and np.size(freqs, axis=0)>1) and freqs[-1]<nyquistfreq:
            hz=freqs
        else:
            raise ValueError("Las frecuencias solo se mantienen como lineales o logaritmicas, vea documentacion.")
    elif (type(freqs)==list or type(freqs)==tuple or type(freqs)==type(np.zeros((0))) and np.size(freqs, axis=0)>1):
        if type(Ncicles)==int:
            Ncicles=np.ones((len(freqs), 1))*Ncicles
            hz=freqs
        elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
            if len(Ncicles)!=len(freqs):
                raise ValueError("El numero de elementos en Ncicles debe ser el mismo que freqs")
            else:
                hz=freqs
  
    t=np.arange(-lentimewav, lentimewav, 1/fsampling)
    nwav=len(t)

    Lconv=len(senal) + nwav-1   #Longitud de la convolucionn
    TFmatrix=np.zeros((len(hz), Lconv), dtype=np.float32)
    #fwhmvec=np.zeros((len(hz)), dtype=np.float32)
    #Comienza el ciclo por frecuencicas
    # print(len(hz))
    for i in range(len(hz)):    
        ## Creacionn de la wavelet de Morlet compleja
        # print("N ciclos: ", Ncicles[i], "Frecuencia: ", hz[i], "duracionn: ", lentimewav)
        mwavelet= WaveletMorlet(fsampling, Ncicles[i], hz[i], t=lentimewav, fwhm=False, rsp=False)
        mwaveletX=fft.fft(mwavelet, Lconv)
        mwaveletX=mwaveletX/max(mwaveletX)
        conv=fft.ifft(fft.fft(senal, Lconv)*mwaveletX)
        TFmatrix[i, :]=np.abs(conv)**2            #Mi¶dulo cuadrado para obtener la potencia.
    return t, hz, TFmatrix[:, nwav//2-1:-1-nwav//2+1:1]




# --------------------------------------------------------------------------------------------------------------    
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
#                                 SPIKE TO LFP METHODS
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
    
    
# Spike-Field Coherence calculated by the spectrum method    
def SFCoherence(n, y, t):                           #INPUT (spikes, fields, time)
    """
    Function written by Sergio Parra Sanchez, 
    
    
    This function calculates the SPike-Field COherence by the spectrum method.
    The function requires the following inputs:
    * n: Signal with the spikes, format is a 2d-numpy array, rows are trials, columns time. 
          The format is a binary matrix (0 no spike, 1 spikes.)
    * y: LFP segment at the same interval than the spikes, the format is the same, trialsXtime
    * t is a numpy array containing the times aligned of all trials. 
    
    
    This function was inspired from the book: 
    
    """
    K = np.shape(n)[0]                          #... where spikes and fields are arrays [trials, time]
    N = np.shape(n)[1]
    dt=t[1]-t[0]
    SYY = np.zeros(int(N/2+1))
    SNN = np.zeros(int(N/2+1))
    SYN = np.zeros(int(N/2+1), dtype=complex)
    
    for k in np.arange(K):
        yf = fft.rfft((y[k,:]-np.mean(y[k,:])) *np.hanning(N))    # Hanning taper the field,
        nf = fft.rfft((n[k,:]-np.mean(n[k,:])))                   # ... but do not taper the spikes.
        SYY = SYY + ( np.real( yf*np.conj(yf) ) )/K                  # Field spectrum
        SNN = SNN + ( np.real( nf*np.conj(nf) ) )/K                  # Spike spectrum
        SYN = SYN + (          yf*np.conj(nf)   )/K                  # Cross spectrum

    cohr = np.real(SYN*np.conj(SYN)) / SYY / SNN                     # Coherence
    f = fft.rfftfreq(N, dt)                                       # Frequency axis for plotting
    
    return (cohr, f, SYY, SNN, SYN)

#Field-Triggered Average function
    
def FTA_function(y, n, t, Wn):                  #INPUTS: y=field, n=spikes, t=time, Wn=passband [low,high]
    K = np.shape(n)[0]                          #... where spikes and fields are arrays [trials, time]
    N = np.shape(n)[1]
    dt = t[1]-t[0]                           #Define the sampling interval.
    fNQ = 1/dt/2                             #Define the Nyquist frequency.
    ord  = 100                               #...and filter order,
    b = sn.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming'); #...build bandpass filter.
    FTA=np.zeros((K, N))                      #Create a variable to hold the FTA.
    for k in np.arange(K):                   #For each trial,
        Vlo = sn.filtfilt(b, 1, y[k,:])  # ... apply the filter.
        phi = sn.angle(sn.hilbert(Vlo))  # Compute the phase of low-freq signal
        indices = np.argsort(phi)            #... get indices of sorted phase,
        FTA[k,:] = n[k, indices]              #... and store the sorted spikes.
    phi_axis = np.linspace(-np.pi, np.pi, N)   #Compute phase axis for plotting.
    return np.mean(FTA, 0), phi_axis    


# def ev_sfc(Wn, fNQ, K, N, ):
    
#     Wn = [44,46]                       # Set the passband
#     b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming');
    
#     del phi
#     phi=zeros([K,N])                # Create variable to hold phase.
#     for k in arange(K):             # For each trial,
#         Vlo = signal.filtfilt(b, 1, y[k,:])       # ... apply the filter,
#         phi[k,:] = angle(signal.hilbert(Vlo))  # ... and compute the phase.
    
#     n_reshaped   = copy(n)
#     n_reshaped   = reshape(n_reshaped,-1)   # Convert spike matrix to vector.
#     phi_reshaped = reshape(phi, -1)         # Convert phase matrix to vector.
#                                                # Create a matrix of predictors [1, cos(phi), sin(phi)]
#     X            = transpose([ones(shape(phi_reshaped)), cos(phi_reshaped), sin(phi_reshaped)])
#     Y            = transpose([n_reshaped])  # Create a vector of responses.
    
#     model = sm.GLM(Y,X,family=sm.families.Poisson())    # Build the GLM model,
#     res   = model.fit()                                 # ... and fit it,
    
#     phi_predict = linspace(-pi, pi, 100)       # ... and evaluate the model results.
#     X_predict   = transpose([ones(shape(phi_predict)), cos(phi_predict), sin(phi_predict)])
#     Y_predict   = res.get_prediction(X_predict, linear='False')
    
#     FTA, phi_axis = FTA_function(y,n,t,Wn)       #Compute the FTA, in the new frequency interval
    
#     plot(phi_axis, FTA)                          #... and plot it, along with the model fit.
#     plot(phi_predict, Y_predict.predicted_mean, 'k')
#     plot(phi_predict, Y_predict.conf_int(), 'k:')
#     xlabel('Phase')
#     ylabel('Probability of a spike');