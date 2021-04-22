import numpy as np
#cimport numpy as cnp
from cython cimport boundscheck, wraparound

def permarray(double[:] array,  double[:, :] MatrizRes):    
    cdef int m, ncol,  nperm, i, j;
    m=len(array);
    nperm=np.size(MatrizRes, axis=1);
    m2=np.size(MatrizRes, axis=0);
    assert m ==m2
    with boundscheck(False), wraparound(False):
        for i in range(nperm):
            np.random.shuffle(array) 
            for j in range(m2):
                MatrizRes[j, i]=array[j]   
    return None


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
### Fúnción para el cálculo de ajustes lineales.
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
cpdef double[:]  linearfit(double[:] xdata, double[:] ydata,  int N):
    """
    Esta función devuelve los parámetros de un ajuste lineal mediante el uso de mínimos cuadrados, la función         
    requiere tres entradas:
    *  xdata: Abscsa de los datos
    *  ydata: Ordenada de los datos
    *  params: arreglo donde se colocarán la pendiente y la ordenada al origen.    
    * N: El número de elementos en cada uno de los arreglos de datos.
    """
    cdef double meanx=0, meany=0, num=0, den=0; 
    cdef int  i=0;
    cdef double[:] params;
    params=np.zeros((2), dtype=np.float64)
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
    return params
    


