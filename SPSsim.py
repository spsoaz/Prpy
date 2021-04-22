import  numpy as np
from numba import njit, float32, float64, int32, int64
import scipy.stats as stats

def poisson_generator(rate, t_i, t_f, n_trials, **kwargs):
    """
        poisson_generator(rate. t_i, t_f, n_triasl)
        
        Esta función genera una matriz de trenes de espigas con una tasa aproximada utilizando:
        N= rate* (t_f*t_i)   donde N es el número de eventos en el intervalo
    
       Ejemplo:
       Creación de 20 trenes con una tasa de 10, el periodo es de 0 a 1000 ms.
       trials=poisson_generator(10, 0, 1000, 20)
       
    """
    N=int((t_f-t_i)*rate)
    spikes=-1/rate * np.log(np.random.rand(n_trials, N )) #Isi
    spikes=(np.cumsum(spikes, axis=1)) + t_i
    return spikes

def simmilar_trains(train, n, lamda):
    """
    simmilar_trains(train)
    
    Esta función devuelve n trenes de espigas similares al primero en el sentido de que son altamente similares en los tiempos de ocurrencia. La similutud se modula con el parámetro lamda, esta lamda funcionará como la sigma de una distribución normal tiempo a tiempo para mover los tiempos de los trenes producidos posteriormente.
    
    Ejemplo:
    Creación de 20 trenes similares considerando una lambda de 10 ms
    
    
    """
    import numpy.random
    import numpy.matlib
    nelem=len(train)
    perturbation= np.random.normal(size=(n, nelem), loc=0, scale=lamda)
    return perturbation + np.matlib.repmat(train, n, 1)

def duty_cycle(rate_l, rate_u, proportion, duration, period):
    """
       duty_cycle(rate_l, rate_u, proportion, trials)
       
       This function returns a duty_cicle of lenght t by using a stepsize of 1 ms.            
       
       example:
       Creation of a duty-cycle of 25 # of work , rate profile is 10 a low state and 30 at high state. Duration of propfile will be 500 ms. The profile begin at low like in sine 
       RateProfile=duty_cycle(rate_l=10, rate_u=30, proportion=0.25, duration=500)
    """
    t=range(0, duration)
    thr=2*proportion-1
    frec=2*np.pi/period
    Rate_profile=((np.sin(frec*np.array(t))+thr)>=0)*(rate_u-rate_l) + rate_l
    return Rate_profile
    
def stim_poisson(RateProfile, ntrials):
    """
        stim_poisson(RateProfile, ntrials)
        
        This function returns the simulation of a stimulus period by using a poisson simulation. To do this, is required 
        a Rate Profile, the function the generates ntrials of such simulation.
        
        This function assumes that the dt intervale in rate profile is 1 ms. Additionaly this function assumes that is treated with a like step-size profile.
                
        **Example:**
        
        trials_stim=stim_poisson(RateProfile, ntrials=20)
    """
    derivative=RateProfile[1::]-RateProfile[0:-1:1]    
    pos=np.where(derivative!=0)[0]  #Extract positions where change occurs
    ti=0
    tf =pos[0]
    trials=poisson_generator(rate=RateProfile[ti+1], t_i=ti, t_f=tf, n_trials=ntrials)
    for i in range(1, len(pos)):
       ti=pos[i-1]
       tf=pos[i]
       trials=np.concatenate((trials, poisson_generator(rate=RateProfile[ti+1], t_i=ti, t_f=tf, n_trials=ntrials)), axis= 1)
    ti=pos[-1]
    tf=len(RateProfile)
    trials=np.concatenate((trials, poisson_generator(rate=RateProfile[ti+1], t_i=ti, t_f=tf, n_trials=ntrials)), axis= 1)       
    return trials

#@njit()
## Generation of Poisson spike train with refractoriness
def  poisson_refr(fr_mean, ns=1000):
    """
        poisson_refr(fr_mean, ns=1000)
        
        This function returns the simulation of a stimulus period by using a poisson simulation. To do this, is required 
        a Rate Profile, the function the generates ntrials of such simulation.
        ti
        
        This function assumes that the dt intervale in rate profile is 1 ms. Additionaly this function assumes that is treated with a like step-size profile.
                
        **Example:**
        
        trials_stim=poisson_refr(fr_mean, ntrials=20)
    """
    fr_mean/=1000;  # mean firing rate per ms.
    ## generating poisson spike train
    lamda=1/fr_mean;   # inverse firing rate
    isi1=-lamda*np.log(np.random.rand(ns,1)); # generation of expo. distr. ISIs
    ## Delete spikes that are within refractory period
    isi=[]
    for i in range(ns):
        if np.random.rand(1)>np.exp(-isi1[i]**2/32):
            isi=isi+list(isi1[i]);
        #end
   # end
    ## Ploting histogram and caclulating cv
   # hist(isi,50);   # Plot histogram of 50 bins
    cv=np.std(isi)/np.mean(isi) # coefficient of variation
    spikes=np.cumsum(isi)/1000 #Para tener el tiempo en s.
    return  spikes, cv

        
def LIF_model(dt=0.1, tau=10, E_L=-65, theta=-55, RI_ext=12):
    """ 
        Simulation of (leaky) integrate-and-fire neuron.
        model parameters:
        dt=0.1
        tau=10
        E_L=-65
        theta=-55
        RI_ext=12
        
        This function returns the voltage, time spikes vector, and time of simulation.
        
        Example:
        v_rec, s_rec, t_rec=LIF_model()
    """ 
    ## Integration with Euler method
    t_step=0
    v=E_L
    time=np.arange(0, 100, dt)
    v_rec=np.zeros((len(time)), dtype=float)
    t_rec=np.zeros((len(time)), dtype=float)
    s_rec=[]
    for t in time:     
        s=v>theta   #boolean value converted to 1 or zero in the following parts
        v=s*E_L+(1-s)*(v-dt/tau*((v-E_L)-RI_ext))
        v_rec[t_step]=v
        t_rec[t_step]=t
        if s>0:
            s_rec.append(t)
        t_step=t_step+1
    return v_rec, s_rec, t_rec



def gammaSpikes( rate, ti, tend):
    """
    Esta función devuelve un tren de espigas cuya distribicuión de 
    intervalos interespigaa es de tipo Gamma. 
    """
    Nspikes=int((tend-ti)*rate)
    ISI=stats.gamma.rvs(1.99,  size=Nspikes)  
    Spikes=(np.cumsum(ISI))+ti
    return Spikes.reshape(Nspikes, 1)


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#                                            Simulactions about LFP 
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

def LFPsim(duration, fsampling):
    """
       This functions simmulates an LFP recording by the proceding described at:
       Rutishauser U, Ross IB, Mamelak AN, Schuman EM. Human memory strength is predicted by
theta-frequency phase-locking of single neurons. Nature. 2010;464:903–7.


       This function has two inputs:
       1.- duration: the seconds of te simmulation
       2.- fsampling: Sampling frewuency of the simulation.

    
    """
    if duration<=0:
        raise ValueError("Duration cannot be negative or zero. See documentation.")
    if fsampling <=0:
        raise ValueError("Duration cannot be negative or zero. See documentation.")

    time=np.arange(0, duration, 1/fsampling)   #Time in seconds
    signal=np.zeros((len(time)), dtype=np.float32)
    for freq in range(1, 101):
        phaseshift=np.random.uniform(0, 2*np.pi)
        amplitude=10/freq
        signal+=amplitude*np.sin(2*np.pi*freq*time + phaseshift)
    return signal + np.random.normal(loc=0, scale=20, size=len(time))
        


def chirpL(duration, fsamp, F1, F2, Amp=1, phase_init=0, **kwargs):
    """
    This function creates a linear chirp of length duration in seconds at a frequency sampling of fsamp.
    
    duration
    fsamp
    F1
    F2
    Amp
    phase_init
    
    """
    t=np.arange(0, duration, 1/fsamp)
    senal=Amp*np.sin(np.pi*t**2*(F2-F1)/duration + 2*np.pi*F1*t + phase_init)
    return senal
















