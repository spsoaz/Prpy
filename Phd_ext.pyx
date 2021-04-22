# -*- coding: utf-8 -*-
"""
Creado por: Sergio Parra Sánchez
Complemento a la biblioteca Phd, en esta se encuentran todas las funciones cython
que permiten optimizar algunas las funciones en la libreria Phd.

El archivo a compilar es Phd_ext_setup.py
"""
from numpy import zeros, mean, std, arange, exp, loadtxt
from numpy import pi, sin, cos, arctan, size, sqrt 
from cpython.array cimport array
#from cython.view cimport array as cvarray
import numpy as np
#cimport numpy as cnp
cimport cython
DTYPE=np.int
DTYPEd=np.double
ctypedef np.int_t DTYPE_t
def SLoadb(str cfile):
     """
     Archivo que carga los datos que empiezan con \t y su separador es \t en la 
     Psicofisica, esto es muy importante ya que indicará que el archivo si es cuadrado
     len
     """
     from numpy import zeros 
     file=open(cfile, 'r')
     Data=zeros((300, 50), dtype=float)
     pos=zeros((2, 1), dtype=int)
     cdef int ind, Llin
     cdef int nrow=0
     for linea in file:
         vec=zeros((500, 1), dtype=float)
         ind=0;
         if linea[0]==" ":
            pos[0, 0]=1
         else:
            pos[0, 0]=0 
         Llin=(len(linea))
         probing=linea.find(',', pos[0, 0], Llin)
         if len(probing)==-1:
             string='\t'
         else:
             string=','
         while (True):
             pos[1, 0]=linea.find(string, pos[0, 0], Llin)
             if pos[1, 0]==1  and linea[0]==' ':
                 pos[0, 0]=2
                 pos[1, 0]=linea.find(string, pos[0, 0], Llin)
             if pos[1, 0]!=pos[0, 0] and pos[1, 0]!=-1:
                 Data[nrow, ind]=float(linea[pos[0, 0]:pos[1, 0]])
                 ind+=1
             elif pos[1, 0]==-1:
                 break;
             pos[0, 0]=pos[1, 0]+1
         nrow+=1    
     return Data[nrow, ind]

def SLoad(str cfile, str delimiter):
    """
    Archivo que carga los datos.
    
    """
    file=open(cfile, 'r')
    Data=[]
    pos=zeros((2, 1), dtype=int)
    cdef DTYPE_t ind, Llin, aux;
    for linea in file.readlines():
        vec=zeros((2400, 1), dtype=float)
        ind=0;
        aux=0;
        while(not linea[aux].isnumeric()):
            aux=aux+1;
        pos[0, 0]=aux
        Llin=int(len(linea))
        while (True):
            pos[1, 0]=linea.find(delimiter, pos[0, 0], Llin)
            if pos[1, 0]!=pos[0, 0] and pos[1, 0]!=-1:
                vec[ind]=float(linea[pos[0, 0]:pos[1, 0]])
                ind+=1
            elif pos[1, 0]==-1:
                Data.append(vec[0: ind])
                break;
            pos[0, 0]=pos[1, 0]+1
    return Data




def sync_file(data, epoch, int nrow,  float period):
    """
    Dada una lista de arreglos como se suelen cargar los archivos de espigas y el inicio y
    final de un intervalo, esta función devuelve el número de cuentas que están en ese intervalo para
    cada elemento de la lista.
    """
    from numpy import array as nparray
    from Phd import vecstrength
    sync=zeros((nrow, 2), dtype=float)
    cdef DTYPE_t length
    for length in range(nrow):
         bolvec=zeros((len(data[length]), 1), dtype=bool)
         bolvec[1::]=(data[length][1::]>=epoch[length, 0])*(data[length][1::]<=epoch[length, 1])        
         theta, bartheta, sync[length, 1], sync[length, 0]=vecstrength(nparray(data[length])[bolvec], period)
#         theta, bartheta, R, vs
    return sync

       

def firing_file(data, double[:, :] epoch, int Nrow, int zt):
    """
    Dada una lista de arreglos como se suelen cargar los archivos de espigas y el inicio y
    final de un intervalo, esta función devuelve el número de cuentas que están en ese intervalo para
    cada elemento de la lista.
    
    zt es un indicador sobre la estandarización de los datos, si zt es igual a 1 se realizará la 
    transformada z dado el periodo anterior a la bajada de la punta.
    """
    cdef DTYPE_t length
    cdef double media=0, sigma=1
    firing=zeros((Nrow, 1), dtype=float)
    if zt==1:
        ztransm=zeros((Nrow, 1), dtype=int)
        for length in range(Nrow):
            bolvec=(data[length][1::]>=0)*(data[length][1::]<=(epoch[length, 1]-epoch[length, 0]))
            ztransm[length]=sum(bolvec)
        media=mean(ztransm)
        sigma=std(ztransm)
    for length in range(Nrow):
        bolvec=(data[length][1::]>=epoch[length, 0])*(data[length][1::]<=epoch[length, 1])
        firing[length]=(sum(bolvec)-media)/sigma
    return firing

def firing(data, double[:, :] epoch, np.int Nrow, np.int zt, double wn, double stp):
    """
    Dada una lista de arreglos como se suelen cargar los archivos de espigas y el inicio y
    final de un intervalo, esta función devuelve el número de cuentas que están en ese intervalo para
    cada elemento de la lista, barriendo por ventana y paso fijo.
    
    zt es un indicador sobre la estandarización de los datos, si zt es igual a 1 se realizará la 
    transformada z dado el periodo anterior a la bajada de la punta, con la misma ventana y tamaño de paso.
    
    Nota: COn el fin de optimizar, esta función devuelve el conteo (cuando no está normalizado zt=0)
    
    """
    cdef DTYPE_t length, count1=np.int((epoch[0][2]-epoch[0][1])/stp), count2
    cdef double media=0, sigma=1, beg
    fir=np.zeros((Nrow, count1), dtype=np.int)
    if zt==1:
        count2=0
        ztransm=zeros((np.int(epoch[0][0]/stp+1)*Nrow+10, 1), dtype=np.int)
        for length in range(Nrow):
            beg=0
            while(beg+wn< epoch[length, 0] ):
                bolvec=(data[length][1::]>=beg)*(data[length][1::]<=(beg+wn))
                ztransm[count2]=np.sum(bolvec)
                count2+=1
                beg+=stp
        media=mean(ztransm[0:count2])
        sigma=std(ztransm[0:count2])
    for length in range(Nrow):
        count1=0
        beg=epoch[length][1]
        while(beg+wn < epoch[length][2]):
            bolvec=(data[length][1::]>=beg)*(data[length][1::]<beg+wn)
            beg=beg+stp
            fir[length][count1]=(np.sum(bolvec))
            count1+=1
    if zt==1:
        return fir, media, sigma   
    return fir



cpdef void firing_file_wind(Data, double[:, :] Tiempos, int nrow, double window, double step, informationfile, file):
    """
     Dada una lista de arreglos como se suelen cargar los archivos de espigas, así como el inicio y final
     de un intervalo para cada ensayo, esta función calculará el firing rate instantáneo utilizando una ventana
     causal. Esta función pondrá los resultados en un archivo con extensión .csv. Para ello el archivo
     debe ser pasado como argumento así como un vector con las características de cada ensayo, las cuales serán los primeras
     columnas de cada archivo. Esta función contempla que estas columnas sean las mismas en número para todos los ensayos.

    Los argumentos de entrada son los siguientes:
        Data
        Tiempos
        nrow
        window
        step
        informationfile
        file
    """
    from numpy import sum as nsum
    from numpy import zeros, array, savetxt, size    
    cdef long int ncol=size(informationfile, axis=1), counter, col, rows
    cdef double window_b, window_e
    for rows in range(nrow):
        window_b=Tiempos[rows, 0]
        window_e=window_b + window
        counter=ncol
        Result=zeros((1, 1200), dtype=float)
        for col in range(ncol):
            Result[0, col]=informationfile[rows, col]
        while(window_e<=Tiempos[rows, 1]):
             bolvec=(array(Data[rows][1::])>=window_b)*(array(Data[rows][1::])<window_e)
             Result[0, counter]=nsum(bolvec)
             window_b+=step
             window_e=window_b+window
             counter+=1
        savetxt(file, Result[0:counter], delimiter=',', newline='\n')
        

cpdef void sync_file_wind(Data, Tiempos, int nrow, float Period, float window, float step, informationfile, file1, file2):
    """
     Dada una lista de arreglos como se suelen cargar los archivos de espigas, así como el inicio y final
     de un intervalo para cada ensayo, esta función calculará el firing rate instantáneo utilizando una ventana
     de media onda causal. Esta función pondrá los resultados en un archivo con extensión .csv. Para ello el archivo
     debe ser pasado como argumento así como un vector con las características de cada ensayo, las cuales serán los primeras
     columnas de cada archivo. Esta función contempla que estas columnas sean las mismas en número para todos los ensayos.

    Los argumentos de entrada son los siguientes:
        Data
        Tiempos
        nrow
        Period
        window
        step
        informationfile
        file1
        file2
    """
    from numpy import zeros, array, savetxt, size
    cdef long int ncol=size(informationfile, axis=1), counter
    cdef double window_b, window_e
    cdef int col, rows
    for rows in range(nrow):
        window_b=Tiempos[rows, 0]
        window_e=window_b + window
        counter=ncol
        VS=zeros((1, 1200), dtype=float)
        Ray=zeros((1, 1200), dtype=float)
        for col in range(ncol):
            VS[0, col]=informationfile[rows, col]
            Ray[0, col]=informationfile[rows, col]
        while(window_e<=Tiempos[rows, 1]):
            bolvec=zeros( (len(Data[rows]), 1), dtype=bool)
            bolvec[1::]=(array(Data[rows])[1::]>=window_b)*(array(Data[rows])[1::]<window_e)
            theta, bartheta, Ray[0, counter],  VS[0][counter]=vecstrength(array(Data[rows])[bolvec], Period)
            window_b+=step
            window_e=window_b+window
            counter+=1
        savetxt(file1, VS[0:counter], delimiter=',', newline='\n')
        savetxt(file2, Ray[0:counter], delimiter=',', newline='\n')
        
        
        
def vecstrength(data, T):
    
    """
     Function created by Sergio Parra,
     this function calculates the vector strength parameter of a vector of times, in the case of single recording
     is a time spikes vector.
     
     To calculate vector strength uses the following formula:
         
         \theta_i=  2\pi *  \frac{ mod(t_i, T) }{ T }
         vs = 1/N * \sqrt{ ( \sum_{i} cos(\theta_i) )^2 + ( \sum_{i} sin(\theta_i) )^2   }
         
         Where N is the spikes number in the train.
         
    Significance is determined by the Rayleigh statistics:
        
        R=2 vs^2  N
        
        Input parameters: 
            
        *1 data, a vector whose elements are the time events.
        *2 Period of interest.
    
    """
    from numpy import pi, sin, cos, arctan, size, sqrt 
    if (size(data)>0):
        theta=(2*pi*(data%T))/T
        sumsin=sum(sin(theta))
        sumcos=sum(cos(theta))
    else:
        return 0, 0, 0, 0
    if(sumcos>0):
        bartheta=arctan( sumsin/sumcos )
    elif (sumcos<0 and sumsin>=0):
        bartheta=pi + arctan(sumsin/sumcos)
    elif (sumcos<0 and sumsin<0):
        bartheta=-pi +arctan(sumsin/sumcos)
    elif ( sumcos==0 and sumsin>0):
        bartheta=pi/2
    elif (sumcos==0 and sumsin<0):
        bartheta=-pi/2
    else:
        bartheta=0    
    vs=1/size(data)*sqrt(sumsin**2 + sumcos**2)
    R=(1-1/(2*size(data)) )*2*vs**2*size(data) + size(data)/2*vs**4
    return theta, bartheta, R, vs        

def vector_strength_file(str cfile,   double[:, :] Intervalos, double period):
    """
    Dado el nombre de un archivo, un vector de tiempos  y el periodo, esta función devolverá un arreglo con n elementos y dos columnas donde se indica el 
    vector de fuerza y luego el estadístico de Rayleigh para cada renglon del archivo. El periodo a tomar en cuenta es el comprendido entre los intervalos
    indicados por los elementos en el vector de tiempos con el cual viene esta función.                 
    """
    
    file=open(cfile, 'r')
    Data=zeros((200, 2), dtype=float)
    pos=zeros((2, 1), dtype=int)
    cdef DTYPE_t    nlinea=0, Llin, countspikes=0
    cdef double value, theta, sumsin, sumcos;
    delimiter=',';
    for linea in file.readlines():        
        pos[0, 0]=2
        countspikes=0;
        sumsin=0;
        sumcos=0;
        Llin=int(len(linea))
        pos[1, 0]=linea.find(delimiter,  pos[0, 0], Llin)
        if pos[1, 0]==-1:
            delimiter='\t'
        while (True):
            pos[1, 0]=linea.find(delimiter, pos[0, 0], Llin)
            if pos[1, 0]!=pos[0, 0] and pos[1, 0]!=-1:
                value=float(linea[pos[0, 0]:pos[1, 0]])
                if value>=Intervalos[nlinea, 0] and value<=Intervalos[nlinea, 1]:
                    value=value-Intervalos[nlinea, 0];
                    theta=((2*pi*value)/period)%(2*pi);
                    sumsin+=sin(theta);
                    sumcos+=cos(theta);
                    countspikes+=1
                elif countspikes>0:
                    break                
            else:
                    countspikes=1;
                    break;
            pos[0,0]=pos[1,0]+1                  
        Data[nlinea, 0]=sqrt(sumsin*sumsin   +  sumcos*sumcos)/countspikes
        Data[nlinea, 1]=(1-1/(2*countspikes) )*2*Data[nlinea, 0]**2*countspikes + countspikes/2*Data[nlinea, 0]**4      
        nlinea+=1
    file.close()
    return Data[0:nlinea, :]




def firing_file_FFT(data, double[:, :] epoch, int Nrow, double wn, double stp):
    """
     Dada una lista de arreglos como se suelen cargar los archivos de espigas, así como el inicio y final
     de un intervalo para cada ensayo, esta función calculará el firing rate instantáneo utilizando una ventana
     causal.
    Los argumentos de entrada son los siguientes:
       data
       epoch
       Nrow
       wn
       stp
    """
    from numpy import sum as nsum
    from numpy import zeros, array    
    cdef long int counter=0, rows, count=0
    cdef double window_b, window_e
    FR=zeros((Nrow, 100), dtype=float) #Se cree que hay 85 valores.1
    zscore=0
    zscoresq=0  #Acumulador de los valores al cuadrado.
    for rows in range(Nrow):
        window_b=0
        window_e=window_b + wn
        while(window_e<=epoch[rows, 0]):
            zscore+=nsum( (array(data[rows][1::])>0)*(array(data[rows][1::])<window_e) )
            zscoresq+=(  nsum((array(data[rows][1::])>0)*(array(data[rows][1::])<window_e) ) )**2
            window_b+=stp
            window_e+=stp
            count+=1
        window_b=epoch[rows, 1]-0.2
        window_e=window_b + wn
        counter=0
        while(window_e<=(epoch[rows, 2]+ 0.2)):
             FR[rows, counter]=nsum( (array(data[rows][1::])>=window_b)*(array(data[rows][1::])<window_e) )     
             window_b+=stp
             window_e+=stp
             counter+=1
    zscoresq=zscoresq/count - (zscore/count)**2 #sigma**2
    zscoresq=zscoresq**(0.5)
    zscore/=count #media
    FR=(FR-zscore)/zscoresq #Tasa de disparo normalizada.
    return FR

cpdef np.ndarray[float, ndim=1] funkernel(double resolution, np.int32_t whker, double fs):
    """
    This frunction calculates the kernel using a sampling of 2khertz.
    Receive a double parameter, the resolution of the kernel and a second parameter
    indicating the kind of kernel acording to the following rule:
    0: Half-wave rectified
    1: Gaussian
    2: Deterministic gaussian
    
    
    """
    cdef DTYPE_t longitud=int(resolution*20*fs )
    #cdef np.ndarray[float, ndim=1] kernel=np.zeros((longitud), dtype=float);
    kernel=np.zeros((longitud), dtype=np.float32);
    if whker==0:
        kernel=((1/resolution)**2)*(np.arange(0, resolution*20, 1/fs)*np.exp((-1/resolution)*(np.arange(0, resolution*20, 1/fs))))
        filename="FRcausalR_"+str(int(resolution*1000))+"fs%d.csv"%(int(fs))
    else:
        print("It 's still in process ")
        return -1;
    kernel=np.flip(kernel)
    #Write data to a file
    file=open(filename, "w")
    np.savetxt(file, kernel, delimiter=",")
    file.close()
    return kernel                

def FRcausalR(double[:] spiketimes, double resolution, double fs):
    """
    Author: Sergio Parra Sánchez
    This function creates the estimate in time of a firing rate dependent of time by
    using a causal half-wave rectification function as a kernel. 
    
    resolution must be in miliseconds.
    fs frequency sampling
    spiketimes is a vector with the event times (positive values) of each action potential
    
    For further information see Peter & Dayan, Theoretical Neuroscience, chapter 1
    
    """
    from numpy import exp, copy, zeros, arange, size
    from math import ceil
    from os import listdir
    ## Para verificar la realización de un archivo y si no, generarlo....
    namefile="FRCausalR_"+str(int(resolution))+"fs%d.csv"%(int(fs)) #"{:.2f}".format(resolution)
    listfiles=listdir()
    if namefile in listfiles:
        kernel=loadtxt(namefile, delimiter=",", dtype=float)
    else:
        kernel=funkernel(resolution,  0, fs) # 0 indicates the half-wave rectification function
    cdef int lk=len(kernel)
    cdef double alpha=1/resolution, stp=1/fs;
    cdef int numel=ceil((spiketimes[-1])*fs);
   # spiketimes=np.subtract(spiketimes, spiketimes[0])# Reset the first time to zero.
    #cdef np.ndarray[DTYPEd, ndim=1] tasa=np.zeros((numel), dtype=DTYPEd);
    tasa=np.zeros((numel), dtype=float);
    cdef int i=0, j;
    cdef int N=len(spiketimes);
    cdef int pos=int(spiketimes[0]*fs);
    while pos<lk: #loop for the first spikes.
        tasa[0:pos]=tasa[0:pos]+kernel[lk-pos:lk];
        i+=1;
        if N>i+1:
            pos=int(spiketimes[i+1]*fs);
        else:
            break
    for j in range(i+1, N):
        pos=int(spiketimes[j]*fs);
        tasa[pos-lk:pos]=tasa[pos-lk:pos]+kernel;
    return tasa





def FRcausalSquared(double[:] spiketimes, double resolution, double fs):
    """
    Author: Sergio Parra Sánchez
    This function creates the estimate in time of a firing rate dependent of time by
    using a deterministic squared function as a kernel. 
    
    resolution must be in seconds.
    fs frequency sampling
    spiketimes is a vector with the event times of each action potential
    
    For further information see Peter & Dayan, Theoretical Neuroscience, chapter 1
    
    """
    from numpy import  zeros, arange, size
    t=arange(spiketimes[0]-resolution, spiketimes[-1], 1/fs, dtype=float);
    cdef int numel=size(t);
    cdef int i, j
    cdef int N=size(spiketimes)
    FRa=zeros((1, numel), dtype=float)
    Id=zeros((1, numel), dtype=bool)
    for i in range(0, N):
        Id=(t<spiketimes[i])*( t>=spiketimes[i]-resolution)
        FRa[Id]+=1
    return FRa/resolution


#function CV2(spike_trains, wnd,  stp, t0, tend)
##wnd=0.3
##stp=0.050
##spike_trains=data;
#r=length(spike_trains);    
#micv2=[];
#mitim=[];
##tend=spike_trains[1][end];
##t0=spike_trains[1][1];
#limit=Int(round((tend-t0)/stp)-wnd/stp+1);
#NTRIAL=zeros(Float64, r, limit);
#NCV2=zeros(Float64, r, limit);
#for i=1:r  #Crea el raster de la cualidad local.

    #ISI=diff(spike_trains[i], dims=1);    
    ##println("ISi calculado");
    #append!(micv2, [2 .*(abs.(ISI[2:end].-ISI[1:end-1]))./(ISI[2:end] .+ISI[1:end-1])]);    
    #append!(mitim, [spike_trains[i][3:end]]);       #Restas 2, 1 por el número de ensayo y otro por diff.  
    ##Ahora comienza el promedio por ensayos.
    ##println("Aquí comienza el promedio por ensayos")
    #ti=0; 
    #column=1
    #while(true)
        #segment=(mitim[i].>=ti).&(mitim[i].<(ti +wnd));      
        #NTRIAL[i, column]=sum(segment);
       ## println(length(segment));
        #NCV2[i, column ]=sum(micv2[i][segment[:]]);
        #ti+=stp;
        #column+=1;
        ##println("Aqui llegué")
        #if mitim[i][end]< (ti+wnd) 
            #break
        #end
    #end
#end
#return NTRIAL, NCV2    
#end
#println("Hecho")    


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
### Fúnción para el cálculo de ajustes lineales.
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
cpdef void  linearfit(double[:] xdata, double[:] ydata, double[:] params):
    """
    Esta función devuelve los parámetros de un ajuste lineal mediante el uso de mínimos cuadrados, la función         
    requiere tres entradas:
    *  xdata: Abscsa de los datos
    *  ydata: Ordenada de los datos
    *  params: arreglo donde se colocarán la pendiente y la ordenada al origen.    
    """
    cdef double meanx=0, meany=0, num=0, den=0; 
    cdef int N=np.size(xdata), i=0;
    params[0]=0
    params[1]=0
    with boundscheck(False), wraparound(False):
        for i in range(N):
            meanx+=xdata[i]
            meany+=ydata[i]
        meany/=N
        meanx/=N
        for i in range(N):
            num+=(ydata[i]*(xdata[i] - meanx)  )
            den+=(xdata[i]*(xdata[i] - meanx)  )          
    params[0]=num/den
    params[1]=meany-meanx*params[0]
    
### Función para el cálculo de tasa de disparo

#def tasa(double[:] spiketimes, double step, 
