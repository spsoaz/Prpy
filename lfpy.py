"""
Módulo escrito por:
Jaime Héctor Osornio
Jerónimo Zizumbo Colunga
Gabriel Díaz de León
Sergio Parra Sánchez

Instituto de Fisiología Celular, Universidad Nacional Autónoma de México


Posiblemente se use el formato nwb neurodata without borders
"""
# Carga libererías
import numpy as np
import scipy.fft as fft
from scipy import signal
import matplotlib.pyplot as plt
#from scipy import stats
#import statsmodels.api as sm
#import numba as nb

#direction="/home/sparra/AENHA/LFP33/LFP_RR033/S1_izq_MAT/RR033077_001"

# Load lfp data
def loadlfp(direction):
    """
    This function loads mat files with the information of lfp. COnsider that this functions could function
    only for data of monkey 32 and 33 due to the conversion made for the functions for the sorting, so probably would be necessary to do another function for lfp loading when we analyze data of other ares of the same monkey or even data from other monkeys.
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
    by Sergio Parra Sánchez.
    
    This function loads and optionaly writes a file with the data contained
    in a nsx file.
    
    """

    import matplotlib.pyplot as plt
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
# Métodos basados en la transformada de Fourier
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# Función para calcular el espectrograma mediante la transformada corta de FOurier

def Spectrogram(data, srate, width, noverlap=0.2, plotting=False, window=[], **kwargs):
    """
       Esta función realiza un espectrograma mediante la técnica de la transformada de Fourier,
       la función utiliza el taper de Hamming exclusivamente hasta el momento
       data: numpy array
       srate: Frecuencia de muestreo, escalar
       width: Ancho de la ventana en segundos
       noverlap: Traslape en fracción [0, 1], default: 0.2 (20%)
       plotting: bool, True muestra el gráfico, default:False
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
        window: Este parámetro es para indicar el tipo de ventana que se utilizará en el proceso
                de la creación del mapa de tiempo frecuencia.Se trata de un vector con la misma longitud 
                que el tamaño de la ventana para ir utilizandolo a modo de taper.
                
       
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
    """

    width_n= np.int32(width*srate/2)   #Ancho en términos del número de puntos dada la frecuencia de muestreo7
    paso=np.int32(np.round(width*srate*(1-noverlap)))
    hz=np.linspace(0, srate/2, width_n-1)
    centimes=np.int32(np.round(np.linspace(width_n, len(data)-width_n, paso)))
    if len(window)==0:
        hammingWin=np.hamming(width_n*2)
    else:
        hammingWin=window
    TFmatrix=np.zeros((len(centimes), len(hz)), dtype=np.float32)
    for tf in range(len(centimes)-1):
        Xdata=hammingWin*data[centimes[tf]-width_n: centimes[tf] + width_n ]
        Xdata=fft.fft(Xdata)
        Xdata=2*np.abs(Xdata[0:len(hz)])/width_n
        TFmatrix[tf, :]=Xdata
    if not plotting:
        return hz, centimes/srate, TFmatrix
    else:
        import matplotlib.pyplot as plt
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
        return hz, centimes/srate, TFmatrix, [fig, ax]

# FFT by the multitaper method using slepian tapers
def fftmultitaper(data, srate, bw):
    """
     Método para el cálculo del espectro de una señal basado en el método de los multiTapers de 
     Thomson. Esta función hace uso de la secuencia slepiana calculada en el módulo de scipy.signal.windows.
     
     Esta función requiere de las siguientes entradas:
     data: Serie de tiempo, un arreglo unidimensional de numpy
     srate: Frecuencia de muestreo
     bw:    Ancho de banda para estimar el número de tapers slepianos ntapers=bw*2-1
     
     Esta función devuelve el espectro de potencia escalado al número de datos.
    """
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
        x = abs(fft.fft(data*temptaper)/n)**2;

        # add this spectral estimate to the total power estimate
        mtPow[:]+= x[0:len(hz)]
    # Because the power spectra were summed over many tapers,
    # divide by the number of tapers to get the average.
    return mtPow[:]/ntap

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# Métodos basados en wavelets
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

def WaveletMorlet(fsamp, Nciclos, f, t=2, fwhm=False, rsp=False, **kwargs):
    """
    Esta función realizará una wavelet de morlet.La wavelet de Morlet se define de la siguiente manera:
    
    $mwv=\frac{1}{\sqrt{s\sqrt{\pi}}} e^{\frac{-t^2}{2s^2}}e^{i2\pi ft}$

    Donde:
    $s=\frac{\#Ciclos}{2\pi f}$.
    
    La función requiere de los siguientes parámetros:
    fsamp: Frecuencia de muestreo
    Nciclos: Número de ciclos
    f: Frecuencia en la cual se centrará.
    t: Opcional, duración temporal, esta se mantiene en dos segundos por default [-2, 2].
    fwhm: Opcional, el ancho espectral como filtro para la señal. Por default es False.
    rsp: Opcional, devuelve el espectro de potencia de la wavelet compleja. Default=False.
    
    [1] Cohen Mike X. Analyzing Neural Time Series Data, Theory and Practice, MIT Press 2014. Chap 13
    
    
    Como ejemplos tiene las siguientes funciones:
    
    ** Ejemplo 1:
        wavelet, fwhm = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=True, rsp=False)
    ** Ejemplo 2:
        wavelet = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=False, rsp=False)
    ** Ejemplo 3:
        wavelet, fwhm, mwaveletX = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=True, rsp=True)
    ** Ejemplo 4:
        mwavelet, mwaveletX = lfp.WaveletMorlet(1000, 3, 10, t=2, fwhm=False, rsp=False)
        
    """
    if t<=0:
        raise ValueError("La duración debe ser positiva, vea la documentación.")
    if Nciclos<=0:
        raise ValueError("Los ciclos deben ser estrictamente positivos, vea la documentación")
    t=np.arange(-t, t, 1/fsamp)
    w=2*(Nciclos/(2*np.pi*f))**2
    mwavelet=np.exp(1j*2*np.pi*f*t) * np.exp((-t**2)/w) #onda compleja * Gaussiana
    if not (np.abs(mwavelet[0])<=0.01 and np.abs(mwavelet[-1])<=0.01):
        raise ValueError("La duración de la wavelet no es suficiente para amortiguar la señal, considere incrementar la duración")
    
    if fwhm: #Devuelve el fwhm
        mwaveletX=fft.fft(mwavelet)
        mwaveletX=mwaveletX*np.conj(mwaveletX)
        mwaveletX/=np.max(mwaveletX)
        freqs=np.linspace(0, fsamp/2, len(mwavelet)//2+1)  #Calcula las frecuencias
        fwhmM=np.argmax(mwaveletX)  #Encuentra la posición del máximo
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
    """
    Esta función realiza el espectrograma de una señal mediante la convolución con wavelets de Morlet complejas,
    la función recibe los siguientes parámetros de entrada:
    """
    nyquistfreq=fsampling/2                  # Frecuencia de Nyquist
    if type(freqs)==str:
        if freqs=="linear": 
            if type(Ncicles)==int:
                hz=np.linspace(1, nyquistfreq-100, len(senal)//4 +1)    # Generación de frecuencias espaciadas linealmente
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
            elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
                hz=np.linspace(1, nyquistfreq-100, len(Ncicles))
        elif freqs=="log":  
            if type(Ncicles)==int:
                hz=np.logspace(np.log10(1), np.log10(nyquistfreq-100), len(senal)//4 +1)  # Generación de frecuencias espaciadas logarítmicamente
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
            elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
                hz=np.logspace(np.log10(1), np.log10(nyquistfreq-100), len(Ncicles))
                Ncicles=np.ones((len(hz)), dtype=np.int32)*Ncicles
        else:
            raise ValueError("Las frecuencias solo se mantienen como lineales o logarítmicas, vea documentación.")
    elif (type(freqs)==list or type(freqs)==tuple or type(freqs)==type(np.zeros((0))) and np.size(freqs, axis=0)>1):
        if type(Ncicles)==int:
            Ncicles=np.ones((len(freqs), 1))*Ncicles
            hz=freqs
        elif (type(Ncicles)==list or type(Ncicles)==tuple or type(Ncicles)==type(np.zeros((0))) and np.size(Ncicles, axis=0)>1):
            if len(Ncicles)!=len(freqs):
                raise ValueError("El número de elementos en Ncicles debe ser el mismo que freqs")
            else:
                hz=freqs
  
    t=np.arange(-lentimewav, lentimewav, 1/fsampling)
    nwav=len(t)

    Lconv=len(senal) + nwav-1   #Longitud de la convolución
    TFmatrix=np.zeros((len(hz), Lconv), dtype=np.float32)
    fwhmvec=np.zeros((len(hz)), dtype=np.float32)
    #Comienza el ciclo por frecuencicas
    print(len(hz))
    for i in range(len(hz)):    
        ## Creación de la wavelet de Morlet compleja
        # print("N ciclos: ", Ncicles[i], "Frecuencia: ", hz[i], "duración: ", lentimewav)
        mwavelet, fwhmvec[i] = WaveletMorlet(fsampling, Ncicles[i], hz[i], t=lentimewav, fwhm=True, rsp=False)
        mwaveletX=fft.fft(mwavelet, Lconv)
        mwaveletX=mwaveletX/max(mwaveletX)
        conv=fft.ifft(fft.fft(senal, Lconv)*mwaveletX)
        TFmatrix[i, :]=2*np.abs(conv)
    return hz, TFmatrix[:, nwav//2-1:-1-nwav//2+1:1], fwhmvec

#--------------------------------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#            METODOS QUE INVOLUCRAN TANTO A ESPIGAS COMO LFP
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
    
    
# Spike-Field Coherence calculated by the spectrum method    
def SFCoherence(n,y,t):                           #INPUT (spikes, fields, time)
    K = shape(n)[0]                          #... where spikes and fields are arrays [trials, time]
    N = shape(n)[1]
    T = t[-1]
    SYY = zeros(int(N/2+1))
    SNN = zeros(int(N/2+1))
    SYN = zeros(int(N/2+1), dtype=complex)
    
    for k in arange(K):
        yf = rfft((y[k,:]-mean(y[k,:])) *hanning(N))    # Hanning taper the field,
        nf = rfft((n[k,:]-mean(n[k,:])))                   # ... but do not taper the spikes.
        SYY = SYY + ( real( yf*conj(yf) ) )/K                  # Field spectrum
        SNN = SNN + ( real( nf*conj(nf) ) )/K                  # Spike spectrum
        SYN = SYN + (          yf*conj(nf)   )/K                  # Cross spectrum

    cohr = real(SYN*conj(SYN)) / SYY / SNN                     # Coherence
    f = rfftfreq(N, dt)                                       # Frequency axis for plotting
    
    return (cohr, f, SYY, SNN, SYN)

#Field-Triggered Average function
    
def FTA_function(y,n,t,Wn):                  #INPUTS: y=field, n=spikes, t=time, Wn=passband [low,high]
    dt = t[1]-t[0]                           #Define the sampling interval.
    fNQ = 1/dt/2                             #Define the Nyquist frequency.
    ord  = 100                               #...and filter order,
    b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming'); #...build bandpass filter.
    FTA=zeros([K,N])                      #Create a variable to hold the FTA.
    for k in arange(K):                   #For each trial,
        Vlo = signal.filtfilt(b, 1, y[k,:])  # ... apply the filter.
        phi = signal.angle(signal.hilbert(Vlo))  # Compute the phase of low-freq signal
        indices = argsort(phi)            #... get indices of sorted phase,
        FTA[k,:] = n[k,indices]              #... and store the sorted spikes.
    phi_axis = linspace(-pi,pi,N)   #Compute phase axis for plotting.
    return mean(FTA,0), phi_axis    
