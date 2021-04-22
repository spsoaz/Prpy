import numpy as np
# -*- coding: utf-8 -*-
"""
Module created by Sergio Parra during Phd, this takes the files after sorting and cleaning.

"""
T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
Psic={"Ntrial":0, "Nclase":1, "Answer":2, "CAnswer":3, "Hit?":4, "ATac1":5, "frec1":6, "AAud1":7, "ATac2":8, "frec2":9, "AAud2":10, "ATac3":11, "frec3":12, "AAud3":13, "RxKd":14,
      "RxKU":15, "RxPB":16, "BT":17, "TRW":18, "Set":19, "Ronda":22}


RR032S1={"serie":0,	"Profundidad":1, 	"Electrodo":2,	"Unidad":3,	"Unitaria":4,	"A":5,	"A_TF":6,	"B":7,	"C":8,	"D":9,
         "E":10, 	"F":11,	"A_Luces":12,	"A_Pasivo":13,	"G":14, "dedo":15, 	"Creceptor":16,  "Protocolo":17, "E_Cuidado":18, "Adaptador":19, "3b":20}


#------------------------------------------------------------------------------------------------------------------------------------------
# Tools for plotting 
mark=['+', 'o', 'v', '^', '*', 'x', '<', '>', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'X', 'D', 'd', '|' , '_']
colorT=['cyan', 'darkturquoise', 'skyblue',  'royalblue', 'blue', 'darkblue', 'navy', 'purple',  'magenta']
colorA=['pink', 'coral', 'darksalmon', 'orange', 'orangered' ,'red', 'crimson', 'firebrick', 'maroon']
#------------------------------------------------------------------------------------------------------------------------------------------

def fitmatlab(x):
    from numpy import log
    return 9.904*log(96.58*x+12.75) + 4.749


class S_RASTER:      
    """
    ******     Created by SERGIO PARRA SANCHEZ
        
    This class creates a set of rasters, raster has all their modalities, in every condition, rows in raster
    are grouped by amplitudes, and in every amplitude, the kind of error or hits during the experiment
    
    This function has four parameters: \n
    *1 cfile: the name of the file to analyze    example="RR032152_002";\n
    *2 electrode: The number of the electrode \n
    *3 unit=2;\n
    *4 align: Indicates the event at which the alignment will occur.\n
    *5 xlim, the limits for the horizontal axis, is an array with 2 elements \n
    *6 TN, Total of neurons sorted
    *7 monkey, the origin of data 32 for RR032 and otherwise RR033
    *8 savef, the path where raster will be saved.
    
    Class has three methods: \n
    
    *1 uncertinty: plots data as in the uncertainty scheme (single align) \n
    *2 focalization: plots data as in the focalization scheme (double align)\n
    *3 attention: plots data with double align and creates two figures, each two subplots \n
    
    """
    # -*- coding: UTF-8 -*-      
    def __init__(self, cfile, electrode, unit, align, xlim, monkey, savef):
        from numpy import size
        self.cfile=cfile
        self.electrode=electrode
        self.unit=unit 
        self.align=align
        self.monkey=monkey
        if(size(xlim)>0):
            self.xlim=xlim 
        else:
            self.xlim=[-5, 3.5] 
        from os import name
        if name=='nt':
            if self.monkey==32:
                self.cpath="D:\BaseDatosKarlitosNatsushiRR032\Text_s"
            else:
                self.cpath="D:\Database_RR033\Text_s"
        else:
            if self.monkey==32:
                self.cpath="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s"
            else:
                self.cpath="/run/media/sparra/AENHA/Database_RR033/Text_s"
        self.savef='%s/%s_%d_%d.png'%(savef, self.cfile, self.electrode, self.unit)
            
    def uncertainty_v2(self):
        """
                    This method plots raster always aligned to the stimulus onset. The plot generated is simmilar to 
                    Yuriria's article. An alternative for alignation is uncertainty method.        

        
        """        
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from Phd_ext import SLoad
        import matplotlib 
        from pyexcel_ods import get_data
        matplotlib.rcParams.update({'font.size': 10, 'font.weight':'bold', 'font.style':'normal'})
        from numpy import loadtxt, unique, copy, arange, size, log10, ceil, zeros
        from numpy import mean as nmin
        if self.monkey==32:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)
        try:
            Psicof=loadtxt(cname, delimiter=","); 
        except:
            Psicof=loadtxt(cname, delimiter="\t"); 
        if self.monkey==32:
            cname='%s/%s/%s_e%d_u%d.csv'%(self.cpath, self.cfile, self.cfile, self.electrode, self.unit)
        else:
            cname='%s/%s/%s_e%d_u%d.csv'%(self.cpath, self.cfile, self.cfile, self.electrode, self.unit)
        try:
            A=loadtxt(cname, usecols=0,  delimiter=',')
        except:
            A=loadtxt(cname, usecols=0,  delimiter='\t')
        A=A-1
        Psicof=copy(Psicof[A.astype(int), :])
        if Psicof[0, 2]==7:
            Psicof[:, 2]=Psicof[:, 3]
        if self.monkey==32:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)
        try:
            Tiempos=loadtxt(cname, delimiter=",")
        except:
            Tiempos=loadtxt(cname, delimiter="\t")
        Tiempos=copy(Tiempos[A.astype(int), :])
        plt.figure(5, figsize=(15, 9))
        plt.title(self.cfile, fontsize=28, fontweight='bold')
        AmpT=unique(Psicof[:, 11])    
        AmpT=copy(AmpT[1::])
        AmpA=unique(Psicof[:, 13])    
        AmpA=copy(AmpA[1::])
        ax=plt.subplot(1, 2, 1)
        ax2=plt.subplot(1, 2, 2)
        ax.yaxis.tick_right()
        ax2.yaxis.tick_right()
        ax.set_xlabel("Time [s]", fontweight='bold', fontsize=12)
        #ax.set_title("Tactile-flutter", fontweight='bold', fontsize=14)
        ax.axis('off')
        #ax2.set_title("Acoustic-flutter", fontweight='bold', fontsize=14)
        ax2.set_xlabel("Time [s]", fontweight='bold', fontsize=12)
        ax2.axis('off')
        step=0.05        
       #Align Parameters
        chcol=zeros((2, 1), dtype=float)
        chcol[0]=nmin(Tiempos[:, 6]-Tiempos[:, 0])/2
        chcol[1]=nmin(Tiempos[:, 9]-Tiempos[:, 0])
        #Plotting confusion of modality, this part, takes into account only such events in which monkey says auditive 
        # when a tactil stimuli arrived 
       
        y=0.0;  
        y2=0.0;
        cname="%s/%s/%s_e%d_u%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data
        Alldata=SLoad(cname, ',')#Loading data
        if len(Alldata[0])==0:
            Alldata=SLoad(cname, '\t')#Loading data
#        cfile='%sS.mat_2_%d_%d.csv'%(self.cfile, self.electrode, self.unit)
        #cfile="%s_e%d_u%d.csv"%(self.cfile, self.electrode, self.unit)
        for amp in AmpT:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==amp)*(Psicof[:, 13]==0)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0)+1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1 #Now reset the count to 0 to coincide with elements.
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, 0]
                chcol[1]=(Tiempos[i, 9]-Tiempos[i, 0]) 
                bchcol=spykes<=chcol[0]                    
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                ax.eventplot(spykes[(bchcol)]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>chcol[1])
                ax.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                ax.eventplot([0], lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                y+=step*1.5 
            if(len(Pos)>0):
                string='- %s'%(str(amp))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)
#        ax.axvspan(xmin=Tiempos[0,  9]-Tiempos[0, 6], xmax=Tiempos[0,  9]-Tiempos[0, 6]+0.1, ymin=0.0, ymax=y, alpha=1, facecolor='gray')
#        ax.axvspan(xmin=Tiempos[0, 10]-Tiempos[0, 6], xmax=Tiempos[0, 10]-Tiempos[0, 6]+0.1, ymin=0.0, ymax=y, alpha=1, facecolor='gray')

        ##Ploting confusion of modality in the acoustic section, that is when monkey says tacitle stimuli but an acoustic stimuli arrived        
        for amp in AmpA:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, 0]              
                bchcol=spykes<=chcol[0]      
                chcol[1]=(Tiempos[i, 9]-Tiempos[i, 0]) 
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                ax2.eventplot(spykes[(bchcol)]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>chcol[1])
                ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)                
                ax2.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                ax2.eventplot([0], lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                y2+=step*1.5
            if(len(Pos)>0):
                string='- %s'%(str(ceil(20.23*log10(amp*37.67)+28.14)))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)
#        #Plotting the cases where not arrived starting in the false alarms        
        y=y+step*5
        if(y2>0):
            y2=y2+step*5
        Order=[3, 1, 2]
        for j in Order:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, 0]
                chcol[1]=(Tiempos[i, 9]-Tiempos[i, 0]) 
                bchcol=spykes<=chcol[0]                                    
                if j==3:     #Monkey answer tactil                      
                    ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                    ax.eventplot(spykes[(bchcol)]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>chcol[1])
                    ax.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='red', linelengths=step, linewidths=1)                
                    ax.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                    ax.eventplot([0], lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                    y+=step*1.5
                elif j==1:   # Monkey answer Auditive
                    ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                    ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>chcol[1])
                    ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)                
                    ax2.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)                    
                    ax2.eventplot([0], lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                    y2+=step*1.5
                else:
                    ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    ax2.eventplot(spykes[(bchcol)], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                    ax.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    ax2.eventplot(spykes[(bchcol)]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>chcol[1])
                    ax.eventplot(spykes[bchcol] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='black', linelengths=step, linewidths=1)                
                    ax.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)                    
                    ax.eventplot([0], lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                    ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='black', linelengths=step, linewidths=1)                                    
                    ax2.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                    ax2.eventplot([0], lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                    y2+=step*1.5
                    y+=step*1.5
        yma=y2
        ymt=y
        ax.annotate('- 0', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))   
        ax2.annotate('- 0', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5)) 
          
 #Plotting tactile stmuli
        Order=[2, 3]
        Color={3:'black',  2:'red'}  

        for ampl in AmpT:
            for j in Order:
                Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                if(len(Pos)>0):
                    for i in Pos:            
                        spykes=Alldata[i][1::]-Tiempos[i, 0] 
                        chcol[1]=(Tiempos[i, 9]-Tiempos[i, 0]) 
                        bchcol=spykes<=(chcol[0])                   
                        ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                        ax.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes>chcol[1])
                        ax.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors=Color[j], linelengths=step, linewidths=1)
                        ax.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                        ax.eventplot([0], lineoffsets=y, colors='gray', linelengths=step, linewidths=0.5)
                        y+=step*1.5
            string='- %s'%(str(ampl))
            ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))   
        ax.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
        ax.yaxis.set_major_formatter(ticker.FixedFormatter((name)))
        ax.set_ylim(0.01, y)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ym2t=y

   #Plotting Auditive stmuli
        Order=[2, 1]
        Color={1:'black', 2:'red'}
        for ampl in AmpA:
            for j in Order:
                Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==ampl)*(Psicof[:, 2]==j) * arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos:            
                    chcol[1]=(Tiempos[i, 9]-Tiempos[i, 0]) 
                    spykes=Alldata[i][1::]-Tiempos[i, 0]
                    bchcol=spykes<=chcol[0]    
                    ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 6]-Tiempos[i, 0]-chcol[0]))*(spykes<=chcol[1])
                    ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>chcol[1])
                    ax2.eventplot(spykes[(bchcol)] -(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors=Color[j], linelengths=step, linewidths=1)                                                     
                    ax2.eventplot(Tiempos[i, 6::]-Tiempos[i, 0]-(Tiempos[i, 6]-Tiempos[i, 0]-2*chcol[0])+ 0.2, lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)
                    ax2.eventplot([0], lineoffsets=y2, colors='gray', linelengths=step, linewidths=0.5)                    
                    y2+=step*1.5  
            string='-%s'%(str(ceil(20.23*log10(ampl*37.67)+28.14)))
            ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))                       
        ym2a=y2
        ax2.yaxis.set_major_locator(ticker.FixedLocator(([25*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y2])))
        name=('0 db', '15 db', '40 db', '54 db', '60 db', '68 db')
        ax2.yaxis.set_major_formatter(ticker.FixedFormatter((name)))
        ax2.set_ylim(-0.01, y2)
        ax2.set_xlim(self.xlim[0], self.xlim[1]) 
       
        ax.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(0, y+step*2), color='gray')   
        ax2.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(0, y2+step*2), color='gray')    
        ax.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.1, y+step*6), color='black', fontsize=8)   
        ax2.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.1, y2+step*6), color='black', fontsize=8)
        ax.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[2, 8]- Tiempos[2, 0]-(Tiempos[2, 6]-Tiempos[2, 0]-2*chcol[0])+ 0.2, y+step*2), color='gray')   
        ax2.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[2, 8]- Tiempos[2, 0]-(Tiempos[2, 6]-Tiempos[2, 0]-2*chcol[0])+ 0.2, y2+step*2), color='gray')    
        ax.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[2, 8]-Tiempos[2, 0]-(Tiempos[2, 6]-Tiempos[2, 0]-2*chcol[0])+ 0.1, y+step*6 ), color='black', fontsize=8)
        ax2.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[2, 8]-Tiempos[2, 0]-(Tiempos[2, 6]-Tiempos[2, 0]-2*chcol[0])+0.1, y2+step*6), color='black', fontsize=8)
        ax.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 9]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]), y+step*2), color='gray')   
        ax2.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 9]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]), y2+step*2), color='gray')    
        ax.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 10]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]), y+step*2), color='gray')
        ax2.annotate("|", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 10]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]), y2+step*2), color='gray')    

        ax.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 9]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]) + 0.05, y+step*6), color='black', fontsize=8) #xytext=(0.41, 0.96)
        ax2.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 9]-Tiempos[0, 0]-(Tiempos[0, 6]-Tiempos[0, 0]-2*chcol[0]) + 0.05, y2+step*6), color='black', fontsize=8 ) #xytext=(0.41, 0.96)
        
        ymt=ymt/ym2t
        ym2=ymt+0.026
        col=0.07            
        Ston=2*chcol[0]+0.2
        Sten=Ston+0.5
        for i in range(0, 26):
                ax.axvspan(xmin=Ston, xmax=Sten, ymin=ymt, ymax=ym2, alpha = col, facecolor='black')
                col+=0.015
                ymt+=0.026
                ym2+=0.026              
        yma=yma/ym2a
        ym2=yma+0.026
        col=0.07
        for i in range(0, 26):
                ax2.axvspan(xmin=Ston, xmax=Sten, ymin=yma, ymax=ym2, alpha = col, facecolor='black')
                col+=0.015
                yma+=0.026
                ym2+=0.026       
        plt.figure(5), plt.subplots_adjust(top=0.95, bottom=0.05, left=0.017, right=0.975, hspace=0.15, wspace=0.12)
        plt.figure(5), plt.tight_layout()  
        plt.savefig(self.savef)
        plt.show()
#######################################################################################################-----------------------------------------------------------------------------------
#######################################################################################################-----------------------------------------------------------------------------------
        
    def focalization(self):   
        """
        This method plots the raster for neurons in the focalization set. Raster is plotted in blocks.
        * First block contains modality confusions ordered by amplitude
        * Second block contains non stimulus block starting at false alarm
        * third block contains cue modality and are ordered in the following form:
            1 cue and not stimuli
            2 cue and crossed modality (ordered by amplitude, first misses and after hits)
            3 cue and correct modality (ordered by amplitude, first misses and after hits)
        
        """         
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib 
        matplotlib.rcParams.update({'font.size': 10, 'font.weight':'bold', 'font.style':'normal'})
        from numpy import loadtxt, unique, copy, arange, size, log, ceil, zeros
        from numpy import mean as nmin
        #from pyexcel_ods import get_data
        from Phd_ext import SLoad
        if self.monkey==32:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)
        try:
            Psicof=loadtxt(cname, delimiter=","); 
        except:
            Psicof=loadtxt(cname, delimiter="\t");
        cname="%s/%s/%s_e%d_u%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data  
        try:
            A=loadtxt(cname, usecols=0, delimiter=',')
        except:
            A=loadtxt(cname, usecols=0, delimiter='\t')
        A=A-1
        Psicof=copy(Psicof[A.astype(int), :])
        if Psicof[0, 2]==7:
            Psicof[:, 2]=Psicof[:, 3]
        if  self.monkey==32:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)            
        Tiempos=loadtxt(cname); 
        Tiempos=copy(Tiempos[A.astype(int), :])
        plt.figure(5, figsize=(15, 9))
        plt.title(self.cfile, fontsize=28, fontweight='bold')
        AmpT=unique(Psicof[:, 11])    
        AmpT=copy(AmpT[1::])
        AmpA=unique(Psicof[:, 13])    
        AmpA=copy(AmpA[1::])
        ax=plt.subplot(1, 2, 1)
        ax2=plt.subplot(1, 2, 2)
        ax.yaxis.tick_right()
        ax2.yaxis.tick_right()
#        ax.set_xlabel("Time [s]", fontweight='bold', fontsize=12)
#        ax.set_title("Tactile-flutter", fontweight='bold', fontsize=14)
        ax.axis('off')
#        ax2.set_title("Acoustic-flutter", fontweight='bold', fontsize=14)
#        ax2.set_xlabel("Time [s]", fontweight='bold', fontsize=12)
        ax2.axis('off')
        step=0.05
        aligna=0; #alignation element
        chcol=zeros((3, 1), dtype=float)
        chcol[0]=nmin(Tiempos[:, 4]-Tiempos[:, 0])/2 #Time elapsed before cue arrives, zero is PD
        chcol[1]=nmin(Tiempos[:, 6]-Tiempos[:, 5])/2 #Time elapsed between cue and principal stimulus arriving
        chcol[2]=nmin(Tiempos[:, 9]-Tiempos[:, 6]) #Average time elapsed between Ku and principal stimulus arriving             
        corte=zeros((2, 1), dtype=float)
        #Plotting confusion of modality, this part, takes into account only such events in which monkey says auditive 
        # when a tactil stimuli arrived.        
        y=0.0;  
        y2=0.0;   # Here the plotting is on confussion of modality maybe unattentional trials: T-T-A
        cname="%s/%s/%s_e%d_u%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data
        #cfile="%s_e%d_u%d.csv"%(self.cfile, self.electrode, self.unit)
        #cname="%s/%s/%sS.mat_2_%d_%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data
        Alldata=SLoad(cname, ",")
        if len(Alldata[0])==0:
            Alldata=SLoad(cname, "\t")
        #Alldata=get_data(cname)#Loading data
        #cfile='%sS.mat_2_%d_%d.csv'%(self.cfile, self.electrode, self.unit)
        for amp in AmpT:
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==amp)*(Psicof[:, 13]==0)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0) + 1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                y+=step*1.5 
            if(len(Pos)>0):
                string='- %s'%(str(amp))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)
        print('Tactile confusion block has been plotted \n')
        ##Ploting confusion of modality in the acoustic section, that is when monkey says tacitle stimuli but an acoustic stimuli arrived        
        for amp in AmpA:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:       # Here the plotting is on confussion of modality maybe unattentional trials: A-A-T     
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                y2+=step*1.5 
            if(len(Pos)>0):
                string='- %s'%(str(ceil(9.904*log(amp*96.58 + 12.75) + 4.749)))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)
#        #This block corresponds to Tactile-No stimuli-Acoustic and Tactile-Acoustic-Acoustic is another kind of unattentional modality confussion     
        if(y2>0):
            y2=y2+step*5
        if(y>0):
            y=y+step*5
        print('Acoustic confusion block has been plotted \n')
    ########### Plotting second block for both (left and right) rasters
            # arrives cue tactile/acoustic and after arrives 1) Nothing and after Crossed modality            
        py=y
        py2=y2
        Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
        Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
        Pos=copy(Pos[1::])
        Pos=Pos-1
        for i in Pos:                         
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                y+=step*1.5                                 
        if(len(Pos)>0):
                string='- %s'%(str(amp))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)
        for amp in AmpA:
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0)+ 1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:                 
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                y+=step*1.5                                 
            if(len(Pos)>0):
                string='- %s'%(str(ceil(9.904*log(96.58*amp + 12.75) + 4.749)))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)
        ##Ploting confusion of modality in the acoustic section, that is when monkey says tacitile stimuli but an acoustic stimuli arrived                
        Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
        Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
        Pos=copy(Pos[1::])
        Pos=Pos-1
        for i in Pos:       # Here the plotting is on confussion of modality maybe unattentional trials: A-A-T                 
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                y2+=step*1.5                            
        if(len(Pos)>0):
                string='- %s'%(str(ceil(9.904*log(amp*96.58 + 12.76) + 4.749)))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)

        for amp in AmpT:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:       # Here the plotting is on confussion of modality maybe unattentional trials: A-A-T     
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                y2+=step*1.5                               
            if(len(Pos)>0):
                string='- %s'%(amp)
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)
#        #This block corresponds to Tactile-No stimuli-Acoustic and Tactile-Acoustic-Acoustic is another kind of unattentional modality confussion     
        if(y2>py2):
            y2=y2+step*5
        if(y>py):
            y=y+step*5  
        print('Second block related to distractions have been plotted \n')
                                
    ############# End of the plotting for second block   
    ############   Plotting third block, No stimuli
        Order=[3, 1, 2]
        for j in Order:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                bchcol=spykes<=chcol[0]                
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]                               
                if j==3:     #Monkey answer tactil                     
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5                                        
                elif j==1:   # Monkey answer Auditive
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5 
                else:
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='black', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5 
                   # Pursuing with the right raster,  
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='black', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5                                                          
        print('No-cue-----No-stimuli block has just been plotted \n')
# plotting tactile cue followed by no stimuli
        Order=[3, 2]
        for j in Order:
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i]-Tiempos[i, aligna]
                bchcol=spykes<=chcol[0]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1] 
                if j==3:     #Monkey answer tactil        
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)                                       
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5           
                elif j==2:
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='black', linelengths=step, linewidths=1)                                        
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5                                                           
        ax.annotate('- 0', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))  
  ## Plotting acoustic cue followed by no-stimuli 
# plotting acoustic cue and after no stimuli
        Order=[1, 2]
        for j in Order:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1)##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i]-Tiempos[i, aligna]
                bchcol=spykes<=chcol[0]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]                 
                if j==1:     #Monkey answer acoustic      
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)                                                          
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5           
                elif j==2:
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='black', linelengths=step, linewidths=1)                                       
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5                                  
        ax2.annotate('- 0', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))  
        print('Cue followed by no-stimuli block has just been plotted \n')
   #Plotting tactile cue followed by acoustic stimuli
        Order=[3, 2]
        Color={3:'red',  2:'black'}  
        for ampl in AmpA:
            for j in Order:
                Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==ampl)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1)
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos:        
                    spykes=Alldata[i]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors=Color[j], linelengths=step, linewidths=1)
                    y+=step*1.5
            if(len(Pos)>0):
                string='- %s'%(str(ceil(9.904*log(ampl*96.58 + 12.75)+4.749)))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))   
#        ax.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
#        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
#        ax.yaxis.set_major_formatter(ticker.FixedFormatter((name)))
        
  #Plotting Acoustic cue followed by tactile stimuli
        Order=[1, 2]
        Color={1:'red',  2:'black'}  

        for ampl in AmpT:
            for j in Order:
                Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos: 
                    spykes=Alldata[i][1::]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors=Color[j], linelengths=step, linewidths=1)
                    y2+=step*1.5  
            if(len(Pos)>0):
                string='- %s'%(str(ampl))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))   
#        ax2.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
#        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
#        ax2.yaxis.set_major_formatter(ticker.FixedFormatter((name)))       
#        ymt=y
#        yma=y2
 #Plotting tactile cue followed by tactile stimuli
        Order=[2, 3]
        Color={3:'black',  2:'red'}  
        for ampl in AmpT:
            for j in Order:
                Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos:  
                    spykes=Alldata[i]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors=Color[j], linelengths=step, linewidths=1)
                    y+=step*1.5
            string='- %s'%(str(ampl))
            ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))   
#        ax.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
#        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
#        ax.yaxis.set_major_formatter(ticker.FixedFormatter((name)))    
   #Plotting acoustic cue followed by Auditive stmuli
        Order=[2, 1]
        Color={1:'black', 2:'red'}
        for ampl in AmpA:
            for j in Order:
                Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==ampl)*(Psicof[:, 2]==j) * arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos:                     
                    spykes=Alldata[i][1::]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors=Color[j], linelengths=step, linewidths=1)
                    y2+=step*1.5                    
            string='-%s'%(str(ceil(9.904*log(ampl*96.58+12.75)+4.749)))
            ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))                       
#        ax2.yaxis.set_major_locator(ticker.FixedLocator(([25*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y2])))
#        name=('0 db', '31 db', '36 db', '46 db', '52 db', '59 db')
#        ax2.yaxis.set_major_formatter(ticker.FixedFormatter((name)))
        ax2.set_ylim(-0.01, y2)
        ax2.set_xlim(self.xlim[0], self.xlim[1]) 
        ax.set_ylim(-0.01, y)
        ax.set_xlim(self.xlim[0], self.xlim[1]) 
        ##
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(0, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(0, y2+step*2), color='gray')  
        ax.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.05, y+step*6), fontsize=10)
        ax2.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.05, y2+step*6), fontsize=10)
        ##    
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y+step*2), color='gray')
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]- Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y2+step*2), color='gray')           
        ax.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-0.05-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y+step*6), fontsize=10)
        ax2.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-0.05-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y2+step*6), fontsize=10)
        #
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.6), y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.6), y2+step*2), color='gray')    
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 10]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 10]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y2+step*2), color='gray')
        ax.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y+step*6), fontsize=10) #xytext=(0.41, 0.96)
        ax2.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y2+step*6), fontsize=10 ) #xytext=(0.41, 0.96)
        ##
        ax.annotate("Cue", xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+ 0.1, y+step*6)) #xytext=(0.41, 0.96)
        ax2.annotate("Cue", xy=(0.8, 0.9), xycoords='data',xytext=(2*chcol[0] + 0.1, y2+step*6), ) #xytext=(0.41, 0.96)
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.2, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.2, y2+step*2), color='gray') 
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.4, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.4, y2+step*2), color='gray') 
        ##
        ax.annotate("Stim.", xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+ 0.6+ 2*chcol[1], y+step*6)) #xytext=(0.41, 0.96)
        ax2.annotate("Stim.", xy=(0.8, 0.9), xycoords='data',xytext=(2*chcol[0] +0.6+ 2*chcol[1], y2+step*6), ) #xytext=(0.41, 0.96)
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.6 +2*chcol[1], y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+2*chcol[1]+0.6, y2+step*2), color='gray')  
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+1.1 +2*chcol[1], y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+2*chcol[1]+1.1, y2+step*2), color='gray')  

#        ym2t=y
#        ym2a=y2
#        ymt=ymt/ym2t
#        ym2=ymt+0.026
#        col=0.07            
#        for i in range(0, 26):
#                ax.axvspan(xmin=0, xmax=0.5, ymin=ymt, ymax=ym2, alpha = col, facecolor='black')
#                col+=0.015
#                ymt+=0.026
#                ym2+=0.026              
#        yma=yma/ym2a
#        ym2=yma+0.026
#        col=0.07
#        for i in range(0, 26):
#            ax2.axvspan(xmin=0, xmax=0.5, ymin=yma, ymax=ym2, alpha = col, facecolor='black')
#            col+=0.015
#            yma+=0.026
#            ym2+=0.026       
        plt.figure(5), plt.subplots_adjust(top=0.95, bottom=0.05, left=0.017, right=0.975, hspace=0.15, wspace=0.12)
        plt.figure(5), plt.tight_layout() 
        plt.savefig(self.savef)
        plt.show()
        
        
    def attention(self):
        """
        This method plots the raster for neurons in the attention with distractors set. Raster is plotted in blocks.
        * First block contains modality confusions ordered by amplitude
        * Second block contains cue modality and are ordered in the following form (beginning at the bottom):
            1 No stimuli (first false alarm and after correct rejections)
            2 Cue and distractor only (ordered by amplitude, first illusions and after hits)
            3 Cue and both stimuli (ordered by amplitude of the main stimuli and after distractor, first misses and after hits)
        
        Remember that this fuction was done for data resultig of offline sorting
        """         
        import matplotlib.pyplot as plt
        #import matplotlib.ticker as ticker
        import matplotlib 
        matplotlib.rcParams.update({'font.size': 10, 'font.weight':'bold', 'font.style':'normal'})
        from numpy import loadtxt, unique, copy, arange, size, log, ceil, zeros
        from numpy import mean as nmin
        #from pyexcel_ods import get_data
        from Phd_ext import SLoad
        if self.monkey==32:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_Psyc.csv" %(self.cpath, self.cfile, self.cfile)            
        try:
            Psicof=loadtxt(cname, delimiter=","); 
        except:
            Psicof=loadtxt(cname, delimiter="\t"); 
        cname="%s/%s/%s_e%d_u%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data  
        try:
            A=loadtxt(cname, usecols=0, delimiter=',')
        except:
            A=loadtxt(cname, usecols=0, delimiter='\t')
        A=A-1
        Psicof=copy(Psicof[A.astype(int), :])
        if Psicof[0, 2]==7:
            Psicof[:, 2]=Psicof[:, 3]
        if self.monkey==32:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)
        else:
            cname="%s/%s/%s_T.csv" %(self.cpath, self.cfile, self.cfile)
        try:
            Tiempos=loadtxt(cname, delimiter=","); 
        except:
            Tiempos=loadtxt(cname, delimiter="\t");
        Tiempos=copy(Tiempos[A.astype(int), :])
        plt.figure(5, figsize=(15, 9))
        plt.title(self.cfile, fontsize=28, fontweight='bold')
        AmpT=unique(Psicof[:, 11])    
        AmpT=copy(AmpT[1::])
        AmpA=unique(Psicof[:, 13])    
        AmpA=copy(AmpA[1::])
        ax=plt.subplot(1, 2, 1)
        ax2=plt.subplot(1, 2, 2)
        ax.yaxis.tick_right()
        ax2.yaxis.tick_right()
        ax.axis('off')
        ax2.axis('off')
        step=0.05
        aligna=0; #alignation element
        chcol=zeros((3, 1), dtype=float)
        chcol[0]=nmin(Tiempos[:, 4]-Tiempos[:, 0])/2 #Time elapsed before cue arrives, zero is PD
        chcol[1]=nmin(Tiempos[:, 6]-Tiempos[:, 5])/2 #Time elapsed between cue and principal stimulus arriving
        chcol[2]=nmin(Tiempos[:, 9]-Tiempos[:, 6]) #Average time elapsed between Ku and principal stimulus arriving             
        corte=zeros((2, 1), dtype=float)
        #Plotting confusion of modality, this part, takes into account only such events in which monkey says auditive 
        # when a tactil stimuli arrived.        
        y=0.0;  
        y2=0.0;   # Here the plotting is on confussion of modality maybe unattentional trials: T-T-A
        cname="%s/%s/%s_e%d_u%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data   
        #cfile="%s_e%d_u%d.csv"%(self.cfile, self.electrode, self.unit)
        #cname="%s/%s/%sS.mat_2_%d_%d.csv" %(self.cpath, self.cfile, self.cfile, self.electrode, self.unit) #Loading data
        Alldata=SLoad(cname, ',')#Loading data
        if len(Alldata[0])==0:
            Alldata=SLoad(cname, '\t')#Loading data
#        cfile='%sS.mat_2_%d_%d.csv'%(self.cfile, sel
        #Alldata=get_data(cname)#Loading data
        #cfile='%sS.mat_2_%d_%d.csv'%(self.cfile, self.electrode, self.unit)
        for amp in AmpT:
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==amp)*(Psicof[:, 13]>0)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of monkey's distractions or simply modality confusion                
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)                
                y+=step*1.5
            if(len(Pos)>0):
                string='- %s$\mu$m-%sdb'%(str(amp),str(ceil(9.904*log(Psicof[i, 13]*96.58 + 12.75) + 4.749)))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)
          #Second part of confussion modality      
        for amp in AmpT:
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==amp)*(Psicof[:, 13]==0)*(Psicof[:, 2]==1) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of monkey's distractions or simply modality confusion                
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)                
                y+=step*1.5
            if(len(Pos)>0):
                string='- %s$\mu$m-%sdb'%(str(amp), str(ceil(9.904*log(Psicof[i, 13]*96.58 + 12.75) + 4.749)))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5), fontsize=6)                
        print('Tactile confusion block has been plotted \n')
        ##Ploting confusion of modality in the acoustic section, that is when monkey says tacitle stimuli but an acoustic stimuli arrived        
        for amp in AmpA:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]>0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:       # Here the plotting is on confussion of modality maybe unattentional trials: A-A-T     
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                y2+=step*1.5 
            if(len(Pos)>0):
                string='- %s-%s$\mu$m'%(str(ceil(9.904*log(amp*96.58 + 12.75) + 4.749)), str(Psicof[i, 11]))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)
##Second part for ploting confusion of modality in the acoustic section, that is when monkey says tactile stimuli but an acoustic stimulus arrived        
        for amp in AmpA:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==amp)*(Psicof[:, 2]==3) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:       # Here the plotting is on confussion of modality maybe unattentional trials: A-A-T     
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                bchcol=spykes<=chcol[0]                  
                ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                # Plotting psychophysics events:
                ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                y2+=step*1.5 
            if(len(Pos)>0):
                string='- %sdb- %s$\mu$m'%(str(ceil(9.904*log(amp*96.58 + 12.75) + 4.749)), str(Psicof[i, 11]))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5), fontsize=6)
#        #This block corresponds to Tactile-No stimuli-Acoustic and Tactile-Acoustic-Acoustic is another kind of unattentional modality confussion     
        if(y2>0):
            y2=y2+step*5
        if(y>0):
            y=y+step*5
        print('Acoustic confusion block has been plotted \n')
########### Plotting second block for both (left and right) rasters: No stimuli in trial -----------------------------------------------------------------------------------------
        Order=[3, 1, 2]
        for j in Order:
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )##Not stimulus, monkey says tactil
            Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
            Pos=copy(Pos[1::])
            Pos=Pos-1
            for i in Pos:            
                spykes=Alldata[i][1::]-Tiempos[i, aligna]
                bchcol=spykes<=chcol[0]                
                corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]                               
                if j==3:     #Monkey answer tactil                     
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='red', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5                                        
                elif j==1:   # Monkey answer Auditive
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='red', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5 
                else:
                   ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='black', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y, colors='gray', linelengths=step, linewidths=1)
                   y+=step*1.5 
                   # Pursuing with the right raster,  
                   bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                   ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                   ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                   bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                   ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='black', linelengths=step, linewidths=1)
                   # Plotting psychophysics events:
                   ax2.eventplot(Tiempos[i, 9::]-Tiempos[i, aligna]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors='gray', linelengths=step, linewidths=1)
                   y2+=step*1.5    
        ax.annotate('- 0$\mu$m-0db', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))                                                       
        ax2.annotate('- 0$\mu$m-0db', xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))
        print('No-cue-----No-stimuli block has just been plotted \n')
        #Continuation with the block, now arrives cue and distractor, main stimulus is not present, tactile component
 #Plotting tactile cue followed by acoustic distractor
        Order=[3, 2]
        Color={3:'red',  2:'black'}  
        for ampl in AmpA:
            for j in Order:
                Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==0)*(Psicof[:, 13]==ampl)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos:        
                    spykes=Alldata[i][1::]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors=Color[j], linelengths=step, linewidths=1)
                    y+=step*1.5
            if(len(Pos)>0):
                string="-0$\mu$m-%sdb"%(str(ceil(9.904*log(ampl*96.58 + 12.75)+4.749)))
                ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5))   
#        ax.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
#        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
#        ax.yaxis.set_major_formatter(ticker.FixedFormatter((name)))        
  #Plotting Acoustic cue followed by tactile stimuli
        Order=[1, 2]
        Color={1:'red',  2:'black'}  

        for ampl in AmpT:
            for j in Order:
                Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]==0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )
                Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                Pos=copy(Pos[1::])
                Pos=Pos-1
                for i in Pos: 
                    spykes=Alldata[i][1::]-Tiempos[i, aligna]
                    corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                    corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                    bchcol=spykes<=chcol[0]                  
                    ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                    ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                    ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                    bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                    ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors=Color[j], linelengths=step, linewidths=1)
                    y2+=step*1.5  
            if(len(Pos)>0):
                string='- %s$\mu$m-0db'%(str(ampl))
                ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))                      
#Pursuing with second block, plotting tactile cue followed by tactile and distractor stimuli
        Order=[2, 3]
        Color={3:'black',  2:'red'}  
        for ampl in AmpT:
            # Sorting now for the amplitude of distractor stimuli
            Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]>0)
            dis=unique(Psicof[Pos, 13])
            for loopdis in dis:
                for j in Order:
                    Pos=(Psicof[:, 8]>0)*(Psicof[:, 10]==0)*(Psicof[:, 11]==ampl)*(Psicof[:, 13]>0)*(Psicof[:, 2]==j) *arange(1, size(Psicof, 0)+ 1 )
                    Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                    Pos=copy(Pos[1::])
                    Pos=Pos-1
                    for i in Pos:  
                        spykes=Alldata[i][1::]-Tiempos[i, aligna]
                        corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                        corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                        bchcol=spykes<=chcol[0]                  
                        ax.eventplot(spykes[bchcol], lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                        ax.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                        ax.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                        ax.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y, colors=Color[j], linelengths=step, linewidths=1)
                        y+=step*1.5
                if(len(Pos)>0):
                    string='- %s$\mu$m-%sdb'%(str(ampl), str(ceil(9.904*log(loopdis*96.58+12.75)+4.749)) )
                    ax.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y-step*1.5)) 
            
#        ax.yaxis.set_major_locator(ticker.FixedLocator(([24*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y])))
#        name=('0 $\mathbf{\mu m}$', '6 $\mathbf{\mu m}$', '8 $\mathbf{\mu m}$', '10 $\mathbf{\mu m}$', '12 $\mathbf{\mu m}$', '24 $\mathbf{\mu m}$')
#        ax.yaxis.set_major_formatter(ticker.FixedFormatter((name)))    
   #Plotting acoustic cue followed by Auditive stmuli
        Order=[2, 1]
        Color={1:'black', 2:'red'}
        for ampl in AmpA:
            #Sorting by tactile distractor amplitude
            Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]>0)*(Psicof[:, 13]==ampl)
            dis=unique(Psicof[Pos, 11])
            for loopdis in dis:
                for j in Order:
                    Pos=(Psicof[:, 8]==0)*(Psicof[:, 10]>0)*(Psicof[:, 11]>0)*(Psicof[:, 13]==ampl)*(Psicof[:, 2]==j) * arange(1, size(Psicof, 0)+1 )
                    Pos=unique(Pos) #Positions of talctile false alarms, 0 does not count                 
                    Pos=copy(Pos[1::])
                    Pos=Pos-1
                    for i in Pos:                     
                        spykes=Alldata[i][1::]-Tiempos[i, aligna]
                        corte[0]=Tiempos[i, 4]-Tiempos[i, aligna]-2*chcol[0]
                        corte[1]=Tiempos[i, 6]-Tiempos[i, 5]-2*chcol[1]
                        bchcol=spykes<=chcol[0]                  
                        ax2.eventplot(spykes[bchcol], lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes<=( chcol[1]+Tiempos[i, 5]-Tiempos[i, 0]) )*(spykes>(Tiempos[i, 4]-Tiempos[i,0]-chcol[0]))
                        ax2.eventplot(spykes[bchcol]-corte[0]+0.2, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes<=( Tiempos[i, 9]-Tiempos[i, 0] ))*(spykes>( Tiempos[i, 6]-chcol[1]-Tiempos[i, 0] ))
                        ax2.eventplot(spykes[bchcol]-corte[1]-corte[0]+0.4, lineoffsets=y2, colors=[0, 0, 0], linelengths=step, linewidths=0.5)
                        bchcol=(spykes>(Tiempos[i, 9]- Tiempos[i, 0]) )                
                        ax2.eventplot(spykes[bchcol]-corte[0]-corte[1]+0.4, lineoffsets=y2, colors=Color[j], linelengths=step, linewidths=1)
                        y2+=step*1.5  
                if(len(Pos)>0):
                    string='- %sdb-%s$\mu$m'%(str(ceil(9.904*log(ampl*96.58+12.75)+4.749)), str(loopdis))
                    ax2.annotate(string, xy=(0.8, 0.9), xycoords='data', xytext=(self.xlim[1]+0.01, y2-step*1.5))                       
#        ax2.yaxis.set_major_locator(ticker.FixedLocator(([25*1.5*step, 35*1.5*step, 45*1.5*step, 55*1.5*step, 65*1.5*step, y2])))
#        name=('0 db', '31 db', '36 db', '46 db', '52 db', '59 db')
#        ax2.yaxis.set_major_formatter(ticker.FixedFormatter((name)))
        ax2.set_ylim(-0.01, y2)
        ax2.set_xlim(self.xlim[0], self.xlim[1]) 
        ax.set_ylim(-0.01, y)
        ax.set_xlim(self.xlim[0], self.xlim[1]) 
        ##
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(0, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(0, y2+step*2), color='gray')  
        ax.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.05, y+step*6), fontsize=10)
        ax2.annotate("PD", xy=(0.8, 0.9), xycoords='data', xytext=(-0.05, y2+step*6), fontsize=10)
        ##    
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y+step*2), color='gray')
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]- Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y2+step*2), color='gray')           
        ax.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-0.05-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y+step*6), fontsize=10)
        ax2.annotate("PU", xy=(0.8, 0.9), xycoords='data', xytext=(Tiempos[0, 8]-0.05-Tiempos[0, 6]+2*chcol[0]+2*chcol[1]+0.6, y2+step*6), fontsize=10)
        #
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.6), y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.6), y2+step*2), color='gray')    
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 10]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 10]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y2+step*2), color='gray')
        ax.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y+step*6), fontsize=10) #xytext=(0.41, 0.96)
        ax2.annotate("MT", xy=(0.8, 0.9), xycoords='data', xytext=(nmin(Tiempos[:, 9]-Tiempos[:, 6]+2*chcol[0]+2*chcol[1]+0.65), y2+step*6), fontsize=10 ) #xytext=(0.41, 0.96)
        ##
        ax.annotate("Cue", xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+ 0.1, y+step*6)) #xytext=(0.41, 0.96)
        ax2.annotate("Cue", xy=(0.8, 0.9), xycoords='data',xytext=(2*chcol[0] + 0.1, y2+step*6), ) #xytext=(0.41, 0.96)
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.2, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.2, y2+step*2), color='gray') 
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.4, y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.4, y2+step*2), color='gray') 
        ##
        ax.annotate("Stim.", xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+ 0.6+ 2*chcol[1], y+step*6)) #xytext=(0.41, 0.96)
        ax2.annotate("Stim.", xy=(0.8, 0.9), xycoords='data',xytext=(2*chcol[0] +0.6+ 2*chcol[1], y2+step*6), ) #xytext=(0.41, 0.96)
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+0.6 +2*chcol[1], y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+2*chcol[1]+0.6, y2+step*2), color='gray')  
        ax.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+1.1 +2*chcol[1], y+step*2), color='gray')   
        ax2.annotate('|', xy=(0.8, 0.9), xycoords='data', xytext=(2*chcol[0]+2*chcol[1]+1.1, y2+step*2), color='gray')        
        plt.figure(5), plt.subplots_adjust(top=0.95, bottom=0.05, left=0.017, right=0.975, hspace=0.15, wspace=0.12)
        plt.figure(5), plt.tight_layout()   
        plt.savefig(self.savef)
        plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------         
#------------------------------------------------------------------------------------------------------------------------       
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                                                
        
def optimumC(monkey, cfile, electrode, unit, return_histogram=0):
    """
    Created on Mon Feb 25 13:35:21 2019
    
    @author: sparra
    
    Esta función realiza la neurometría basada en el criterio óptimo. Como parámetros de entrada requiere:
    * 1 El nombre del archivo, ej RR032152_001
    * 2 El número del electrodo, ej 4
    * 3 El canal, ej 2
    * 4 opcional, 0 indica que no se regrese el histograma con área igual a 1, otro indica que si.

    Esta función hace uso de la función de psicofísica, la cual devuelve los resultados de psicofísica del día, para ellos
    sólo requiere del nombre del archivo.

    Esta función devuelve una lista con la neurometría, el primer elemento tiene una matriz con la iformación de la neurometría 
    táctil, el segundo elemento contiene la neurometría auditiva y el tercero el resultado del conteo del fr para cada modalidad.

    Nota importante: Esta función solo está diseñada para los sets de ncertidumbre.

    """
    from pyexcel_ods import get_data
    from numpy import zeros, loadtxt, size, unique, sum, size, argmax, max, min, array, linspace, abs, histogram


    ### Information to set as input parameters, until now this is a script
    #path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text"
    if monkey==32:
        #path="D:\\BaseDatosKarlitosNatsushiRR032\\Text_s"
        path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s"
    else:
        path="/run/media/sparra/AENHA/Database_RR033/Text_s"
    numbins=20
    window=0.050 # Tiempo en segundos
    step=0.001 #Tiempo en segundos
    cname="%s/%s/%s_Psyc.csv"%(path, cfile, cfile)
    try:
        Psicof=loadtxt(cname, usecols=(4,11,13), delimiter='\t')
    except:
         Psicof=loadtxt(cname, usecols=(4,11,13), delimiter=',')
    cname="%s/%s/%s_T.csv"%(path, cfile, cfile)
    try:
        Tiempos=loadtxt(cname, usecols=(5,6,7), delimiter='\t')
    except:
        Tiempos=loadtxt(cname, usecols=(5,6,7), delimiter=',')
    espifile="%s/%s/%s"%(path, cfile, cfile) +"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"
    espigas=get_data(espifile, delimiter=',')
    espigas=espigas[cfile+"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"]
    if type(espigas[0][0])==str:
         espigas=get_data(espifile, delimiter='\t')
         espigas=espigas[cfile+"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"]
         if type(espigas[0][0])==str:
             print("No fue posible leer el archivo ", espifile)
             return -1
    try:
        A=loadtxt(espifile, usecols=(0), delimiter=',')
    except:
        A=loadtxt(espifile, usecols=(0), delimiter='\t')
    A=A-1
    Psicof=Psicof[A.astype(int), :]
    Tiempos=Tiempos[A.astype(int), :]
    Numtrials=size(Psicof,0)
    tmp=sum((Psicof[:,1]==0)*(Psicof[:,2]==0))
    FRateZero=zeros((tmp, 2),dtype=float);
    tmp=(Psicof[:,1]>0)*(Psicof[:,2]==0)
    TacAmplitudes=unique(Psicof[tmp, 1])
    tmp=sum(Psicof[:,1]==TacAmplitudes[0])
    listTac=[zeros((tmp, 2))]
    for i in range(1, size(TacAmplitudes)):
        tmp=sum(Psicof[:, 1]==TacAmplitudes[i])
        listTac.append(zeros((tmp, 2)))    
    tmp=(Psicof[:,1]==0)*(Psicof[:,2]>0)
    AudAmplitudes=unique(Psicof[tmp,2])
    tmp=sum(Psicof[:,2]==AudAmplitudes[0])
    listAud=[zeros((tmp, 2))]
    for i in range(1, size(AudAmplitudes)):
        tmp=sum(Psicof[:, 2]==AudAmplitudes[i])
        listAud.append(zeros((tmp, 2)))
    countZ=int(0)
    countTac=zeros((len(TacAmplitudes),1), dtype=int)
    countAud=zeros((len(AudAmplitudes),1), dtype=int)
    mean=0.0
    std=0.0
    counter=0
    for i in range(0, Numtrials):
        beg=0
        while(True):
            tmp=sum(   (array(espigas[i][1::])>=beg)*(array(espigas[i][1::])<=(beg + window) ))
            mean+=tmp
            std+=tmp**2
            beg+=step
            counter+=1
            if (beg+window)>Tiempos[i, 0]:
                break
        maximum=0.0
        minimum=500.0
        beg=Tiempos[i, 1];
        while(True):
            f=sum( (array(espigas[i][1::])>=beg)*(array(espigas[i][1::])<=(beg+window)))/window
            if(f>maximum):
                maximum=f
            if(f<minimum):
                minimum=f
            beg+=step
            if((beg+window)>Tiempos[i, 2]):
                break;            
        if(Psicof[i,1]==0 and Psicof[i,2]==0): #Not stimulus present
            FRateZero[countZ, 0]=maximum
            FRateZero[countZ, 1]=minimum
            countZ+=1
        elif(Psicof[i,1]==0 and Psicof[i,2]>0):  #Auditive stimulus        
            for stim in range(0, size(AudAmplitudes, 0)):
                if (AudAmplitudes[stim]==Psicof[i, 2]):
                   (listAud[stim])[countAud[stim], 0]=maximum
                   (listAud[stim])[countAud[stim], 1]=minimum
                   countAud[stim]+=1
                   break        
        else:
            for stim in range(0, len(TacAmplitudes)):  #Tactile modality 
                if (TacAmplitudes[stim]==Psicof[i, 1]):
                   (listTac[stim])[countTac[stim], 0]=maximum
                   (listTac[stim])[countTac[stim], 1]=minimum                 
                   countTac[stim]+=1
                   break     
    mean=(mean/window)/counter   #mean firing rate before probe down
    std=(std/(window**2))/counter-mean**2  #standard deviation of firing rate before probe down
    if std==0:
        std=1.0
    #Extract the maximum firing rate 
    maximum=max(FRateZero) #first getting of maximum
    tmp=-20 #Set a new temporal maximum at an impossible rate
    for i in range(0, len(listTac)):
        tmp=max(listTac[i])
        if tmp>maximum:
            maximum=tmp
    for i in range(0, len(listAud)):
        tmp=max(listAud[i])
        if tmp>maximum:
            maximum=tmp   
      #Extract the minimum firing rate 
    minimum=min(FRateZero) #first getting of maximum
    tmp=2000 #Set a new temporal maximum at an impossible rate
    for i in range(0, len(listTac)):
        tmp=min(listTac[i])
        if tmp<minimum:
            minimum=tmp
    for i in range(0, len(listAud)):
        tmp=min(listAud[i])
        if tmp<minimum:
            minimum=tmp
    maximum=(maximum-mean)/std
    minimum=(minimum-mean)/std
    X=linspace(minimum-0.5, maximum+0.5, numbins) 
    binwidth=(X[1]-X[0])
    Histograms=zeros( (numbins-1, 1+len(listAud)+len(listTac)), dtype=float)
    #Starting with histogram of zeros
    FRateZero=(FRateZero-mean)/std
    #Identifying neuron as incresing rate or decreasing in function of stimuli amplitude
    listTac[-1]=(listTac[-1]-mean)/std
    if max(abs( (listTac[stim-1])[:, 0])) > max(abs((listTac[stim-1])[:, 1])):
        dim=0
    else:
        dim=1
    Histograms[:, len(TacAmplitudes)], X=histogram(listTac[-1][:, dim], bins=X, density=True)
    Histograms[:, len(TacAmplitudes)]*=binwidth
    for stim in range(1, len(TacAmplitudes)):
        listTac[stim-1]=(listTac[stim-1]-mean)/std
        Histograms[:, stim], X=histogram(listTac[stim-1][:, dim], bins=X, density=True)
        Histograms[:, stim]*=binwidth
    Histograms[:, 0], X=histogram(FRateZero[:, dim], bins=X, density=True)
    Histograms[:, 0]*=binwidth

    #Histogram for Auditive stimuli
    listAud[-1]=(listAud[-1]-mean)/std
    if max(abs( (listAud[stim-1])[:, 0])) > max(abs((listAud[stim-1])[:, 1])):
        dim=0
    else:
        dim=1
    Histograms[:, -1], X=histogram(listAud[-1][:, dim], bins=X, density=True)
    Histograms[:, -1]*=binwidth
    for stim in range(len(AudAmplitudes)-1):
        listAud[stim]=(listAud[stim]-mean)/std
        Histograms[:, stim+ len(TacAmplitudes)+1], X=histogram(listAud[stim][:, dim], bins=X, density=True)
        Histograms[:, stim+len(TacAmplitudes)+1]*=binwidth
    # Aquí empieza el cálculo del criterio óptimo para la neurometría táctil
    Hits=zeros((numbins-1, 1), dtype=float)
    for co in range(1, numbins-1):
        for amp in range(0, 6):
            if amp==0:
                Hits[co]+=(sum(Histograms[0:co, amp] ) )*countZ/Numtrials
            else:
                Hits[co]+=(sum(Histograms[co::, amp] ) )*countTac[amp-1]/Numtrials    

    #        divide[i, 0]=trapz(Histograms[:, i], x=X, dx=binwidth, axis=-1)   #Make integrals
    #Here begin the calculation for Optimum criterion 
    c=argmax(Hits)
    neur=zeros((1, 6), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur[0, amp]+=(sum(Histograms[:c, amp] ) )
        else:
            neur[0, amp]+=(sum(Histograms[c::, amp] ) )    
    neur[0, 0]=1-neur[0, 0]  #Probabilidad de falla para la amplitud cero.
# Until here, was calculated neurometric by using only two possible responses            
            
    Hits2=zeros((numbins-1, 1), dtype=float)
    for co in range(2, numbins-1):
        for amp in [0, 6, 7, 8, 9, 10]:
            if amp==0:
                Hits2[co]+=(sum(Histograms[1:co, amp] ) )*countZ/Numtrials
            else:
                Hits2[co]+=(sum(Histograms[co::, amp] ) )*countAud[amp-6]/Numtrials 
    #Hits2=Hits2/10  
    c2=argmax(Hits2)
    neur2=zeros((1, 6), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur2[0, amp]+=(sum(Histograms[1:c2+1, amp] ) )
        else:
            neur2[0, amp]+=(sum(Histograms[c2+1::, amp + 5] ) )  
    neur2[0, 0]=1-neur2[0, 0]
    if return_histogram==0:        
        return  neur, neur2
    else:
        return neur, neur2, Histograms

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def optimumC_VS(monkey, cfile, electrode, unit, whset):
    """
    Created on Mon Feb 25 13:35:21 2019
    
    @author: sparra
    
    Esta función realiza la neurometría basada en el criterio óptimo. Como parámetros de entrada requiere:
    * 1 cfile:     El número de serie, ej 164
    * 2 electrode: El número del electrodo, ej 4
    * 3 unit       El número de unidad en ese electrodo/canal
    * 4 whset      El número de set como identificador (3 para el set B (incertidumbre))

    Esta función hace uso de la función de psicofísica, la cual devuelve los resultados de psicofísica del día, para ellos
    sólo requiere del nombre del archivo.

    Esta función devuelve una lista con la neurometría, el primer elemento tiene una matriz con la iformación de la neurometría 
    táctil, el segundo elemento contiene la neurometría auditiva y el tercero el resultado del conteo del fr para cada modalidad.

    Nota importante: Esta función solo está diseñada para los sets de incertidumbre.
    Esta función utiliza el archivo producido por la función fir_Inf_pfile cuando la métrica es la del vector de fuerza.

    """
    import numpy as np
    from numpy import zeros, unique, histogram_bin_edges
    from numpy import sum as nsum
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import matplotlib
    from pyexcel_ods import get_data
    matplotlib.rcParams.update({'font.size': 13, 'font.weight':'bold', 'font.style':'normal'})

    ### Information to set as input parameters, until now this is a script
    #path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text"
    if monkey==32:
        path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s"
        #path="D:\\Drive_sparra\\RS2\\"   
    else:
        path="/run/media/sparra/AENHA/Database_RR033/Text_s"        
    cname=path+"%d_"%(monkey)+str(cfile)+"_full_iE30_fE30_sync.csv"
    data=np.loadtxt(cname, delimiter=',')
    mask=(data[:, 1]==electrode)*(data[:, 2]==unit)*(data[:, 6]==whset)
    data=data[mask, :] #Extracción de los datos
    data[:, -2]=data[:, -1]#*data[:,-1]
    edges=histogram_bin_edges(data[:, -2], bins='auto')
    numbins=len(edges)-1
    Histograms=zeros((numbins, 11), dtype=float)
    Numtrials=np.size(data,0)
    tmp=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 9]==0)*(data[:, 10]==0)*(data[:, -2]>0)
    countZ=int(nsum(tmp)) 
    VSZero=data[tmp, -2]
    mask=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 9]>0)*(data[:, 10]==0)*(data[:, -2]>0)
    TacAmplitudes, countTac=unique(data[mask, 9], return_counts=True)
    mask=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 9]==0)*(data[:, 10]>0)*(data[:, -2]>0)
    AudAmplitudes, countAud=unique(data[mask, 10], return_counts=True)
    amplitudes=np.concatenate(([0], TacAmplitudes, AudAmplitudes))
    for i in range(len(TacAmplitudes)+len(AudAmplitudes)+ 1):
        if i==0:
            mask=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 9]==0)*(data[:, 10]==0)
        elif i<=len(TacAmplitudes):
            mask=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 9]==amplitudes[i])*(data[:, 10]==0)
        else:
            mask=(data[:, 7]==0)*(data[:, 8]==0)*(data[:, 10]==amplitudes[i])*(data[:, 9]==0)
        Histograms[:, i], edges=histogram(data[mask, -2], bins=edges, density=True)
    Histograms=Histograms*(edges[1]-edges[0])
    # Aquí empieza el cálculo del criterio óptimo para la neurometría táctil
    Hits=np.zeros((numbins, 1), dtype=float)
    for co in range(1, numbins):
        for amp in range(0, 6):
            if amp==0:
                Hits[co]+=(np.sum(Histograms[0:co, amp] ) )*countZ/Numtrials
            else:
                Hits[co]+=(np.sum(Histograms[co::, amp] ) )*countTac[amp-1]/Numtrials    

    #        divide[i, 0]=np.trapz(Histograms[:, i], x=X, dx=binwidth, axis=-1)   #Make integrals
    #Here begin the calculation for Optimum criterion 
    c=np.argmax(Hits)
    tmp=(Hits==Hits[c])
    tmp=np.multiply(tmp[:, 0], np.arange(0, np.size(Hits, 0) ) )
    c=int(np.max(tmp))
    neur=np.zeros((6, 1), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur[amp]+=(np.sum(Histograms[:c+1, amp] ) )
        else:
            neur[amp]+=(np.sum(Histograms[c+1::, amp] ) )    
# Until here, was calculated neurometric by using only two possible responses            
            
    Hits2=np.zeros((numbins, 1), dtype=float)
    for co in range(2, numbins):
        for amp in [0, 6, 7, 8, 9, 10]:
            if amp==0:
                Hits2[co]+=(np.sum(Histograms[1:co, amp] ) )*countZ/Numtrials
            else:
                Hits2[co]+=(np.sum(Histograms[co::, amp] ) )*countAud[amp-6]/Numtrials 
    #Hits2=Hits2/10  
    c2=np.argmax(Hits2)
    tmp=(Hits2==Hits2[c2])
    tmp=np.multiply(tmp[:, 0], np.arange(0, np.size(Hits2, 0) ) )
    c2=int(np.max(tmp))
    neur2=np.zeros((6, 1), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur2[amp]+=(np.sum(Histograms[1:c2+1, amp] ) )
        else:
            neur2[amp]+=(np.sum(Histograms[c2+1::, amp + 5] ) )         
            
    return  neur, neur2,  Histograms

#------------------------------------------------------------------------------------------------------------------------ ----------------------------------  
#------------------------------------------------------------------------------------------------------------------------ ---------------------------------- 




def optimumC_FFT(monkey, cfile, electrode, unit, whset):
    """   
    @author: sparra
    
    Esta función realiza la neurometría basada en el criterio óptimo. Como parámetros de entrada requiere:
    * 1 cfile:     El número de serie, ej 164
    * 2 electrode: El número del electrodo, ej 4
    * 3 unit       El número de unidad en ese electrodo/canal
    * 4 whset      El número de set como identificador (3 para el set B (incertidumbre))

    Esta función hace uso de la función de psicofísica, la cual devuelve los resultados de psicofísica del día, para ellos
    sólo requiere del nombre del archivo.

    Esta función devuelve una lista con la neurometría, el primer elemento tiene una matriz con la iformación de la neurometría 
    táctil, el segundo elemento contiene la neurometría auditiva y el tercero el resultado del conteo del fr para cada modalidad.

    Nota importante: Esta función solo está diseñada para los sets de incertidumbre.
    Esta función utiliza el archivo producido por la función fir_Inf_pfile cuando la métrica es la del vector de fuerza.

    """
    import numpy as np
    from numpy import zeros, unique, histogram_bin_edges, where
    from numpy import sum as nsum
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import matplotlib
    from pyexcel_ods import get_data
    matplotlib.rcParams.update({'font.size': 13, 'font.weight':'bold', 'font.style':'normal'})
    from sys import name
    from Phd_ext import FRcausalR
    from scipy import fftpack
    ### Information to set as input parameters, until now this is a script
    #path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text"
    if name=="nt":
       path="D:\\BaseDatosKarlitosNatsushiRR032\\Text_s"
    else:
        if monkey==32:
        #path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text"
            path="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s"
        else:
            path="/run/media/sparra/AENHA/Database_RR033/Text_s"            
    numbins=10
    window=0.050 # Tiempo en segundos
    step=0.001 #Tiempo en segundos
    extremos=0.100 # Tiempo en segundos
    cname="%s/%s/%s_Psyc.csv"%(path, cfile, cfile)
    try:
        Psicof=loadtxt(cname, usecols=(4, 11, 13), delimiter='\t')
    except:
         Psicof=loadtxt(cname, usecols=(4, 11, 13), delimiter=',')
    cname="%s/%s/%s_T.csv"%(path, cfile, cfile)
    try:
        Tiempos=loadtxt(cname, usecols=(0, 6, 7), delimiter='\t')
    except:
        Tiempos=loadtxt(cname, usecols=(0, 6, 7), delimiter=',')
    espifile="%s\\%s\\%s"%(path, cfile, cfile) +"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"
    espigas=get_data(espifile, delimiter=',')
    espigas=espigas[cfile+"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"]
    if type(espigas[0][0])==str:
         espigas=get_data(espifile, delimiter='\t')
         espigas=espigas[cfile+"_e"+ str(electrode) + "_u" + str(unit)+ ".csv"]
         if type(espigas[0][0])==str:
             print("No fue posible leer el archivo ", espifile)
             return -1
    try:
        A=loadtxt(espifile, usecols=(0), dtype=int, delimiter=',')
    except:
        A=loadtxt(espifile, usecols=(0), dtype=int, delimiter='\t')
    A=A-1
    Psicof=Psicof[A, :]
    Tiempos=Tiempos[A, :]
    Numtrials=size(Psicof,0)
    tmp=sum((Psicof[:,1]==0)*(Psicof[:,2]==0))
    FRateZero=zeros((tmp, 2),dtype=float);
    tmp=(Psicof[:,1]>0)*(Psicof[:,2]==0)
    TacAmplitudes=unique(Psicof[tmp, 1])
    tmp=sum(Psicof[:,1]==TacAmplitudes[0])
    listTac=[zeros((tmp, 2))]
    for i in range(1, size(TacAmplitudes)):
        tmp=sum(Psicof[:, 1]==TacAmplitudes[i])
        listTac.append(zeros((tmp, 2)))    
    tmp=(Psicof[:,1]==0)*(Psicof[:,2]>0)
    AudAmplitudes=unique(Psicof[tmp,2])
    tmp=sum(Psicof[:,2]==AudAmplitudes[0])
    listAud=[zeros((tmp, 2))]
    for i in range(1, size(AudAmplitudes)):
        tmp=sum(Psicof[:, 2]==AudAmplitudes[i])
        listAud.append(zeros((tmp, 2)))
    countZ=int(0)
    countTac=zeros((len(TacAmplitudes),1), dtype=int)
    countAud=zeros((len(AudAmplitudes),1), dtype=int)
    mean=0.0
    std=0.0
    counter=0
    countbasal=0
    fs=2000  #round(1024/0.7)    
    for i in range(0, Numtrials):
        Tasa=FRcausalR(espigas[i][1::], 0.010, fs)
        limit=round(Tiempos[i, 0]*fs)-1
        mean+=nsum(Tasa[0:limit])
        std+=nsum((Tasa[0:limit])**2)
        counterbasal+=limit
        freqs=fftpack.fftfreq(1024, 1/fs)
        limit=round(Tiempos[i, 1]*fs-0.100)-1
        fft=fftpack.fft(Tasa[limit:limit+1024 ])
        power=(abs(fft)**2)
        
        if(Psicof[i,1]==0 and Psicof[i,2]==0): #Not stimulus present
            FRateZero[countZ, 0]=maximum
            FRateZero[countZ, 1]=minimum
            countZ+=1
        elif(Psicof[i,1]==0 and Psicof[i,2]>0):  #Auditive stimulus        
            for stim in range(0, size(AudAmplitudes, 0)):
                if (AudAmplitudes[stim]==Psicof[i, 2]):
                   (listAud[stim])[countAud[stim], 0]=maximum
                   (listAud[stim])[countAud[stim], 1]=minimum
                   countAud[stim]+=1
                   break        
        else:
            for stim in range(0, len(TacAmplitudes)):  #Tactile modality 
                if (TacAmplitudes[stim]==Psicof[i, 1]):
                   (listTac[stim])[countTac[stim], 0]=maximum
                   (listTac[stim])[countTac[stim], 1]=minimum                 
                   countTac[stim]+=1
                   break     
    mean=(mean/window)/counter   #mean firing rate before probe down
    std=(std/(window**2))/counter-mean**2  #standard deviation of firing rate before probe down
    if std==0:
        std=1.0
    #Extract the maximum firing rate 
    maximum=max(FRateZero) #first getting of maximum
    tmp=-20 #Set a new temporal maximum at an impossible rate
    for i in range(0, len(listTac)):
        tmp=max(listTac[i])
        if tmp>maximum:
            maximum=tmp
    for i in range(0, len(listAud)):
        tmp=max(listAud[i])
        if tmp>maximum:
            maximum=tmp   
      #Extract the minimum firing rate 
    minimum=min(FRateZero) #first getting of maximum
    tmp=2000 #Set a new temporal maximum at an impossible rate
    for i in range(0, len(listTac)):
        tmp=min(listTac[i])
        if tmp<minimum:
            minimum=tmp
    for i in range(0, len(listAud)):
        tmp=min(listAud[i])
        if tmp<minimum:
            minimum=tmp
    maximum=(maximum-mean)/std
    minimum=(minimum-mean)/std
    X=linspace(minimum-0.5, maximum+0.5, numbins) 
    binwidth=(X[1]-X[0])
    Histograms=zeros( (numbins-1, 1+len(listAud)+len(listTac)), dtype=float)
    #Starting with histogram of zeros
    FRateZero=(FRateZero-mean)/std
    #Identifying neuron as incresing rate or decreasing in function of stimuli amplitude
    listTac[-1]=(listTac[-1]-mean)/std
    if max(abs( (listTac[stim-1])[:, 0])) > max(abs((listTac[stim-1])[:, 1])):
        dim=0
    else:
        dim=1
    Histograms[:, len(TacAmplitudes)], X=histogram(listTac[-1][:, dim], bins=X, density=True)
    Histograms[:, len(TacAmplitudes)]*=binwidth
    for stim in range(1, len(TacAmplitudes)):
        listTac[stim-1]=(listTac[stim-1]-mean)/std
        Histograms[:, stim], X=histogram(listTac[stim-1][:, dim], bins=X, density=True)
        Histograms[:, stim]*=binwidth
    Histograms[:, 0], X=histogram(FRateZero[:, dim], bins=X, density=True)
    Histograms[:, 0]*=binwidth

    #Histogram for Auditive stimuli
    listAud[-1]=(listAud[-1]-mean)/std
    if max(abs( (listAud[stim-1])[:, 0])) > max(abs((listAud[stim-1])[:, 1])):
        dim=0
    else:
        dim=1
    Histograms[:, -1], X=histogram(listAud[-1][:, dim], bins=X, density=True)
    Histograms[:, -1]*=binwidth
    for stim in range(len(AudAmplitudes)-1):
        listAud[stim]=(listAud[stim]-mean)/std
        Histograms[:, stim+ len(TacAmplitudes)+1], X=histogram(listAud[stim][:, dim], bins=X, density=True)
        Histograms[:, stim+len(TacAmplitudes)+1]*=binwidth
    # Aquí empieza el cálculo del criterio óptimo para la neurometría táctil
    Hits=zeros((numbins-1, 1), dtype=float)
    for co in range(1, numbins-1):
        for amp in range(0, 6):
            if amp==0:
                Hits[co]+=(sum(Histograms[0:co, amp] ) )*countZ/Numtrials
            else:
                Hits[co]+=(sum(Histograms[co::, amp] ) )*countTac[amp-1]/Numtrials    

    #        divide[i, 0]=trapz(Histograms[:, i], x=X, dx=binwidth, axis=-1)   #Make integrals
    #Here begin the calculation for Optimum criterion 
    c=argmax(Hits)
    neur=zeros((1, 6), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur[0, amp]+=(sum(Histograms[:c, amp] ) )
        else:
            neur[0, amp]+=(sum(Histograms[c::, amp] ) )    
    neur[0, 0]=1-neur[0, 0]  #Probabilidad de falla para la amplitud cero.
# Until here, was calculated neurometric by using only two possible responses            
            
    Hits2=zeros((numbins-1, 1), dtype=float)
    for co in range(2, numbins-1):
        for amp in [0, 6, 7, 8, 9, 10]:
            if amp==0:
                Hits2[co]+=(sum(Histograms[1:co, amp] ) )*countZ/Numtrials
            else:
                Hits2[co]+=(sum(Histograms[co::, amp] ) )*countAud[amp-6]/Numtrials 
    #Hits2=Hits2/10  
    c2=argmax(Hits2)
    neur2=zeros((1, 6), dtype=float)
    for amp in range(0, 6):
        if amp==0:
            neur2[0, amp]+=(sum(Histograms[1:c2+1, amp] ) )
        else:
            neur2[0, amp]+=(sum(Histograms[c2+1::, amp + 5] ) )  
    neur2[0, 0]=1-neur2[0, 0]
    if return_histogram==0:        
        return  neur, neur2
    else:
        return neur, neur2, Histograms

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------- 

class MODULATION:
    """
    Function created by Sergio Parra,
    this function plots the histogram of the Rayleigh statistic and the plot
    of the strenght vector and its angle. To indicate if the neuron has any synchronization, the R statistic must
    surpass the value 13.8 generated by a chi-squared distribution with two degrees of freedom and  P<0.001 
     
    To calculate vector strength uses the following formula:
         
         \theta_i=  2\pi *  \frac{ mod(t_i, T) }{ T }
         r = 1/N * \sqrt{ ( \sum_{i} cos(\theta_i) )^2 + ( \sum_{i} sin(\theta_i) )^2   }
         
         Where N is the spikes number in the train.
         
    Significance is determined by the Rayleigh statistics:
        
        R=( 1- 1/(2N) )*2*N*r^2 + Nr^4/2
        
        Input parameters: 
            
        *1 file, a string with the name of the file. Example RR032167_002
        *2 Number of electrode of interest
        *3 Number of unit of interest.
        *4 Period to take as basis of synchronization 
        *5 Time to analyse, an array with the initial time and end time
            ** Particularly at this, the parameter could be an integer number (input spykes aligned ), [0, 2]
            ** or can be the name of periods as in psychophysycs: ['S3',  'F3']
        *6 typem, indicates the kind of set: 0 uncertainty, otherwise attention (considered as focalization or distractors)
            
            
        Neither this class nor its methods are applicable to the distractors setup, consider expand functionality
    """
    from numpy import loadtxt, size, arange, unique, zeros, ones
    
    def __init__(self, cfile, electrode, unit, Period, Time, typem):   
        from os import name
        if name=='nt':
            self.path="D:\BaseDatosKarlitosNatsushiRR032"
        else:
            self.path = "/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032"
        self.cfile=cfile
        self.electrode=electrode
        self.unit=unit
        self.Period=Period
        self.Time=[0, 0]
        self.typem=typem
        if len(Time)>2:
            raise Exception("Time is not a proper value, see documentation \n")
        if ( (type(Time[0])==float or type(Time[0])==int) and (type(Time[1])==float or type(Time[1])==int)  ): 
            self.Time=Time
        elif ( type(Time[0])==str and type(Time[1])==str ):
            for j in [0, 1]:
                if(Time[j]=="PD"):
                    self.Time[j]=0;
                elif(Time[j]=="KD"):
                    self.Time[j]=1;
                elif(Time[j]=="E1"):
                    self.Time[j]=2;
                elif(Time[j]=="F1"):
                    self.Time[j]=3;
                elif(Time[j]=="E2"):
                    self.Time[j]=4;
                elif(Time[j]=="F2"):
                    self.Time[j]=5;            
                elif(Time[j]=="S3"):
                    self.Time[j]=6;
                elif(Time[j]=="F3"):
                    self.Time[j]=7;
                elif(Time[j]=="PU"):
                    self.Time[j]=8;
                elif(Time[j]=="KU"):
                    self.Time[j]=9;
                elif(Time[j]=="PB"):
                    self.Time[j]=10;
        else:
            raise Exception("Time is not a proper value, see documentation \n")    
            
            
    def Sychronization(self):
        
        """
            This method measures the values for data according to the vector strength
            and the Rayleigh statistic, This method returns the vector strength, 
            the barycenter and the Rayleigh statistic by amplitude.
            
            This method uses the amplitude of the principal stimulus, this function
            is not applicable to the distractors method, consider expand functionality
            
            This function returns an array with 11 columns, one for each amplitude, and
            7 rows in the following order:
                * 1 Amplitude
                * 2 Mean of vector strength per rep
                * 3 Standard deviation of vector strength per rep
                * 4 Mean of varycenter per rep
                * 5 Standard deviation of varycenter per rep
                * 6 Mean of the Rayleigh statistic per rep.
                * 7 Standard deviation of Rayleygh statistic per rep.
                
            This method returns the following results:
              Ray: An array with the Rayleigh statistic obtained per trial, the first column represents results for cue epoch and 
                   the following for main stimulus epoch
              VS:  An array with the vector strength obtained per trial, the first column represents results for cue epoch and 
                   the following for main stimulus epoch  
         classes:  An array with the following structure
                    AcueT     AcueA     AstimT       AstimA      VSmean_cue     VSstd_cue      Ray_mean_cue     Ray_std_cue
                    VSmean_stim      VSstd_stim       Ray_mean_stim        Ray_std_ stim      
              
        """
        from numpy import loadtxt, unique, zeros, mean, std, copy, arange, array
        from numpy import sum as nsum
        from pyexcel_ods import get_data
        cname="%s/Text_s/%s/%s_T.csv"%(self.path, self.cfile, self.cfile)
        Tiempos=loadtxt(cname, dtype=float)
        cname="%s/Text_s/%s/%s_Psyc.csv"%(self.path, self.cfile, self.cfile)
        Psicof=loadtxt(cname, usecols=(8, 10, 11, 13), dtype=float)
        cname="%s/Text_s/%s/%s_e%d_u%d.csv"%(self.path, self.cfile, self.cfile, self.electrode, self.unit)
        A=loadtxt(cname, dtype=int, usecols=(0), delimiter=',')
        A=A-1
        Psicof=copy(Psicof[A, :])
        Tiempos=copy(Tiempos[A, :])
        data=get_data(cname, delimiter=',')
        cname="%s_e%d_u%d.csv"%(self.cfile, self.electrode, self.unit)
        data=data[cname]
        clas=unique(Psicof, axis=0)
        classes=zeros((len(clas), 12), dtype=float)
        classes[:, 0:4]=unique(Psicof, axis=0)
        del clas
        # Reserva espacio para VS y para el vector de Rayleigh
        VS=zeros((len(Psicof), 2), dtype=float)
        Ray=copy(VS)
        for i in range (len(Psicof)):
            find=unique((data[i][:]>=Tiempos[i, 4])*(data[i][:]<=Tiempos[i, 5])*arange(1, len(data[i])+1))[1::]
            find=find-1
            if len(find)>0:
                borrame1, borrame2, Ray[i, 0], VS[i, 0]=vecstrength((array(data[i]))[find], self.Period)   
            else:
                Ray[i, 0]=-1
                VS[i, 0]=-1
            find=unique((data[i][:]>=Tiempos[i, 6])*(data[i][:]<=Tiempos[i, 7])*arange(1, len(data[i])+1))[1::]
            find=find-1
            if len(find)>0:
                borrame1, borrame2, Ray[i, 1], VS[i, 1]=vecstrength((array(data[i]))[find], self.Period)             
            else:
                Ray[i, 0]=-1
                VS[i, 0]=-1
        for i in range(len(classes)):  
            find=(Psicof[:, 0]==classes[i, 0])*(Psicof[:, 1]==classes[i, 1])*(Psicof[:, 2]==classes[i, 2])*(Psicof[:, 3]==classes[i, 3])*(VS[:, 0]>=0)
            classes[i, 4]=mean(VS[find, 0])
            classes[i, 5]=std(VS[find, 0])
            classes[i, 6]=mean(Ray[find, 0])
            classes[i, 7]=std(Ray[find, 0])
            classes[i, 8]=mean(VS[find, 1])
            classes[i, 9]=std(VS[find, 1])
            classes[i, 10]=mean(Ray[find, 1])
            classes[i, 11]=std(Ray[find, 1])
   
        return Ray, VS, classes
            
            
            
            
    def FRmodulation(self):
        
        """
            This method measures the firing rate by the count approximation and after measures
            the mean and standard deviation. 
            
            This method returns an array with 11 columns, one for each amplitude, and
            2 rows in the following order:
                * 1 Amplitude
                * 2 Mean of firing rate per rep
                * 3 Standard deviation of firing rate per rep                
        """
        from numpy import loadtxt, size, arange, unique, zeros, mean, std, copy, array, append
        from pyexcel_ods import get_data
        cname="%s/Text_s/%s/%s_T.csv"%(self.path, self.cfile, self.cfile)
        Tiempos=loadtxt(cname, dtype=float)
        cname="%s/Text_s/%s/%s_Psyc.csv"%(self.path, self.cfile, self.cfile)
        if self.typem==0:
            Psicof=loadtxt(cname, usecols=(4, 11, 13), dtype=float)
        else:
            Psicof=loadtxt(cname, usecols=(4, 8, 10, 11, 13), dtype=float)
        cname="%s/Text_s/%s/%s_e%d_u%d.csv"%(self.path, self.cfile, self.cfile, self.electrode, self.unit)
        A=loadtxt(cname, usecols=0, dtype=int, delimiter=',')
        A=A-1
        Psicof=copy(Psicof[A, :])
        Tiempos=copy(Tiempos[A, :])
        ztrans=0        
        ztrans2=0
        cname="%s/Text_s/%s/%s_e%d_u%d.csv"%(self.path, self.cfile, self.cfile, self.electrode, self.unit)
        spykes=get_data(cname)
        del cname
        cname="%s_e%d_u%d.csv"%(self.cfile, self.electrode, self.unit)
        spykes=spykes[cname]
        begin=0
        for i in range(0, size(Psicof, 0)):            
            ztrans+=(sum(array(spykes[i][1::])<0.1))/0.1
            ztrans2+=( (sum(array(spykes[i][1::])<0.1))/0.1 )**2
        ztrans=ztrans/size(Psicof, 0)
        ztrans2=(ztrans2/size(Psicof, 0)- (ztrans)**2)**(0.5)
    # Sorting by modality
        if self.typem==0:
            Tactile=Psicof[:,1]>0
            Auditive=Psicof[:,2]>0
            Zero=(unique((Psicof[:,1]==0)*(Psicof[:,2]==0)*arange(1, size(Psicof, 0)+1)))[1::]
            Zero=Zero-1
            AmpTac=unique(Psicof[Tactile,1])
            AmpAud=unique(Psicof[Auditive,2])        
            Results=zeros((3, 11))
            r=zeros((200,1))    #firing rate    
            #Not stimulus
            for i in Zero:           
                spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                r[begin]=(r[begin]-ztrans)/ztrans2
                begin+=1
            Results[:, 0]=(0, mean(r[0:begin-1, 0]), std(r[0:begin-1, 0]) )
            # Tactile stimuli
            j=1
            for amp in AmpTac:
                Tactile=( unique((Psicof[:, 1]==amp)*arange(1, size(Psicof, 0)+1)) )[1::]
                Tactile=Tactile-1
                begin=0
                for i in Tactile:
                    spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                    tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                    r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                    r[begin]=(r[begin]-ztrans)/ztrans2
                    begin+=1
                Results[:, j]=(amp, mean(r[0:begin-1, 0]), std(r[0:begin-1, 0]) )
                j+=1
         # Acoustic stimuli               
            for amp in AmpAud:
                Auditive=( unique((Psicof[:, 2]==amp)*arange(1, size(Psicof, 0)+1)) )[1::]
                Auditive=Auditive-1            
                begin=0
                for i in Auditive:
                    cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(self.path, self.cfile, self.cfile, i+2, self.electrode, self.unit)
                    spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                    tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                    r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                    r[begin]=(r[begin]-ztrans)/ztrans2
                    begin+=1
                Results[:, j]=(amp, mean(r[0:begin-1, 0]), std(r[0:begin-1, 0]) )
                j+=1       
            return Results 
        else:   # For focalization and/or Distractors
            Tactile=(Psicof[:, 1]>0)*(Psicof[:, 4]>=0)*(Psicof[:, 3]>=0)
            Auditive=(Psicof[:,2]>0)*(Psicof[:, 3]>=0)*(Psicof[:, 4]>=0)
            Zero=(unique((Psicof[:, 1]==0)*(Psicof[:, 2]==0)*arange(1, size(Psicof, 0)+1)))[1::]
            Zero=Zero-1
            QAmpTac=unique(Psicof[Tactile, 1])   #tact cue ampl 
            QAmpAud=unique(Psicof[Auditive,2])   #aud cue ampl 
            AmpTac=unique(Psicof[Tactile, 3])   
            append(AmpTac, [0])
            AmpAud=unique(Psicof[Auditive, 4])   
            append(AmpTac, [0])
            Results=zeros((5, 30))
            r=zeros((200, 1))    #firing rate    
            #Not stimulus
            for i in Zero:           
                spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                r[begin]=(r[begin]-ztrans)/ztrans2
                begin+=1
            Results[:, 0]=(0, 0, 0, mean(r[0: begin-1, 0]), std(r[0:begin-1, 0]) )
            # Tactile stimuli
            j=1
            for qamp in QAmpTac:
                distractors=( unique(Psicof[ ((Psicof[:, 1]==qamp)*(Psicof[:, 3]>=0)*(Psicof[:, 4]>=0)), 4] )      )
                for ampd in distractors:   #Loop that moves on amplitude distractors                
                    for amp in AmpTac:
                        Tactile=( unique((Psicof[:, 1]==qamp)*(Psicof[:, 3]==amp)*(Psicof[:, 4]==ampd)*arange(1, size(Psicof, 0)+1)) )[1::]
                        Tactile-=1
                        begin=0
                        for i in Tactile:  #For that goes per trial
                            spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                            tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                            r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                            r[begin]=(r[begin]-ztrans)/ztrans2
                            begin+=1
                        if len(Tactile)>0:
                            Results[:, j]=(qamp, ampd, amp, mean(r[0:begin-1, 0]), std(r[0:begin-1, 0]) )
                            j+=1
         # Acoustic stimuli  
            for qamp in QAmpAud:             
                distractors=( unique(Psicof[(Psicof[:, 2]==qamp)*(Psicof[:, 4]>=0)*(Psicof[:, 3]>=0) , 3]))
                for ampd in distractors:
                    for amp in AmpAud:
                        Auditive=( unique((Psicof[:, 2]==qamp)*(Psicof[:, 4]==amp)*(Psicof[:, 3]==ampd)*arange(1, size(Psicof, 0)+1)) )[1::]
                        Auditive=Auditive-1            
                        begin=0
                        for i in Auditive:
                            cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(self.path, self.cfile, self.cfile, i+2, self.electrode, self.unit)
                            spykes[i][1::]=spykes[i][1::]-Tiempos[i, self.Time[0]]
                            tmp=( array(spykes[i][1::])>0.0 )*( array(spykes[i][1::])<( Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]] )  )
                            r[begin]=sum(tmp)/(Tiempos[i, self.Time[1]]- Tiempos[i, self.Time[0]])
                            r[begin]=(r[begin]-ztrans)/ztrans2
                            begin+=1
                        if len(Auditive)>0:
                            Results[:, j]=(qamp, ampd, amp, mean(r[0:begin-1, 0]), std(r[0:begin-1, 0]) )
                            j+=1       
            return Results[:, 0:j] 

    def FRate_st(self):
        
        """
            This method measures the standardized firing rate, the output is an array with the values of the 
            firing rate per amplitude. This method calculate mean firing rate during S3 per trial.
            Method returns all the measurements during the stimulation period 'S3'
            Resultant matrix has the following columns: 
            * 1 Amplitude of stimulus at S2
            * 2 Amplitude of stimulus at S3 (Only in attention or focalization sets)
            * 3 Standardized time average firing rate during S3
            * 4 Standardized maximum (or minimum) firing rate during S3
        """
        from numpy import loadtxt, size, arange, unique, zeros, ceil, max, min, trapz, sqrt
        cname="%s/Text/%s/%s.mat_35.csv"%(self.path, self.cfile, self.cfile)
        Tiempos=loadtxt(cname, usecols=(1, self.Time[0], self.Time[1]), dtype=float)
        cname="%s/Text/%s/%s.mat_1.csv"%(self.path, self.cfile, self.cfile)
        Psicof=loadtxt(cname, usecols=(8, 10, 11, 13), dtype=float)  
        if self.typem==0:
            Results=zeros((size(Tiempos, 0), 3), dtype=float)
        else:
            Results=zeros((size(Tiempos, 0), 4), dtype=float)
         # Stimuli
        med=0
        st=0
        for trial in range(0, size(Tiempos, 0)):
            cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(self.path, self.cfile, self.cfile, trial+2, self.electrode, self.unit)
            spykes=loadtxt(cname, dtype=float)
            Firing=FRcausalR(spykes, 0.050, 1000) # This fR(t) begin at zero in steps of 50 ms
            begin=int(ceil(Tiempos[trial, 1]/0.050)) #Begin of stimulation period
            end=int(ceil(Tiempos[trial, 2]/0.050 ) +1) #end of stimulation period
            #Here begins time average
            media=1/((end-1-begin)*0.050)*trapz(Firing[begin:end], x=None, dx=0.050)
            if self.typem==0:
                Results[trial, 0]=Psicof[trial, 2]+ Psicof[trial, 3]
            else:
                Results[trial, 0]=Psicof[trial, 1] + Psicof[trial, 0]
                Results[trial, 1]=Psicof[trial, 2]+ Psicof[trial, 3]
            Results[trial, -2]=media 
            begin=int(ceil(Tiempos[trial, 0]+0.050/0.050)) #Kd
            end=int(ceil((Tiempos[trial, 0] + 0.550)/0.050 ) + 1) #Kd + 500 ms
            norm=1/((end-1-begin)*0.050)*trapz(Firing[begin: end], x=None, dx=0.050)
            med+=norm
            st+=norm**2
            M=max(Firing[begin: end])
            m=min(Firing[begin: end])
            if M>=-1*m:
                Results[trial, -1]=M
            else:
                Results[trial, -1]=m
        # Calculating mean and standard deviation of non stimulus period
        med=med/size(Psicof, 0)
        st=sqrt(st/size(Psicof, 0)- med**2)
        #Normalizing results
        Results[:, -1]=(Results[:, -1]-med)/st
        Results[:, -2]=(Results[:, -2]-med)/st
        return Results 
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ ----------------------------------   
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------     
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
    else:
        bartheta=0    
    vs=1/size(data)*sqrt(sumsin**2 + sumcos**2)
    R=(1-1/(2*size(data)) )*2*vs**2*size(data) + size(data)/2*vs**4
    return theta, bartheta, R, vs

#------------------------------------------------------------------------------------------------------------------------ ----------------------------------   
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------     
def RVStr(data, T):
    
    """
     Function created by Sergio Parra,
     this function calculates the resonator vector strength function of a vector of times, in the case of single recording
     is a time spikes vector.
     
     To calculate vector strength uses the following formula:
         
         \theta_i=  2\pi *  \frac{ mod(t_i, T) }{ T }
         r = 1/N * \sqrt{ ( \sum_{i} cos(\theta_i) )^2 + ( \sum_{i} sin(\theta_i) )^2   }
         
         Where N is the spikes number in the train.
         
    Significance is determined by the Rayleigh statistics:
        
        R=2 r^2  N
        
        Input parameters: 
            
        *1   data:   A vector whose elements are the time events.
        *2   T       An array including the periods to analiyze.
    
    """
    from numpy import pi, sin, cos, arctan, size, sqrt, zeros
    sizedat=size(data, axis=0)
    if (sizedat>0):
        Nvec=size(T)
        if Nvec>1:
            RVS=zeros((Nvec, 1), dtype=float)
            Ray=zeros((Nvec, 1), dtype=float)
            bartheta=zeros((Nvec, 1), dtype=float)
            for i in range(0, Nvec):
                theta=(2*pi*(data%T[i]))/T[i]
                sumsin=sum(sin(theta))
                sumcos=sum(cos(theta))
                if(sumcos>0):
                    bartheta[i]=arctan( sumsin/sumcos )
                else:
                    bartheta[i]=0 
                RVS[i]=1/sizedat*sqrt(sumsin**2 + sumcos**2)
                Ray[i]=(1-1/(2*sizedat) )*2*RVS[i]**2*sizedat + sizedat/2*RVS[i]**4
            return bartheta, RVS, Ray
        else:
             theta=(2*pi*(data%T))/T
             sumsin=sum(sin(theta))
             sumcos=sum(cos(theta))
             if(sumcos>0):
                 bartheta=arctan( sumsin/sumcos )
             else:
                bartheta=0 
             RVS=1/sizedat*sqrt(sumsin**2 + sumcos**2)
             Ray=(1-1/(2*sizedat) )*2*RVS**2*sizedat + sizedat/2*RVS**4
             return bartheta, RVS, Ray
    else:
        return 0, 0, 0
    
    
#__________________________________________________________________________________________________________________    
#__________________________________________________________________________________________________________________
#__________________________________________________________________________________________________________________
 

def synchronization(file, electrode, unit, T, path):
    """
    Function created by Sergio Parra Sánchez,
    this function plots the histogram of the Rayleigh statistic and the plot
    of the strenght vector and its angle. To indicate if the neuron has any synchrorinzation, the R statistic must
    surpass the value 13.8 generated by a chi-squared distribution with two degrees of freedom and  P<0.001 
     
    To calculate vector strength uses the following formula:
         
         \theta_i=  2\pi *  \frac{ mod(t_i, T) }{ T }
         r = 1/N * \sqrt{ ( \sum_{i} cos(\theta_i) )^2 + ( \sum_{i} sin(\theta_i) )^2   }
         
         Where N is the spikes number in the train.
         
    Significance is determined by the Rayleigh statistics:
        
        R=2 r^2  N
        
        Input parameters: 
            
        *1 file:      A string with the name of the file. Example RR032167_002
        *2 electrode: Number of electrode of interest
        *3 unit:      Number of unit of interest.
        *4 T:         Period to take as basis of synchronization 
        *5 path:      It is the path where we can find data.
    """
    import numpy as np
    import matplotlib.pyplot as plt    ## Begin function   
    cname="%s/Text/%s/%s_T.csv"%(path, file, file)
    Tiempos=np.loadtxt(cname, dtype=float, )
    cname="%s/Text/%s/%s_Psyc.csv"%(path, file, file)
    Psicof=np.loadtxt(cname, usecols=(4, 11, 13), dtype=float)
    rep=np.size(Psicof, 0)
    # Sorting by modality
    Tactile=Psicof[:,1]>0
    Auditive=Psicof[:,2]>0
    Zero=(Psicof[:,1]==0)*(Psicof[:,2]==0)
    Hits=Psicof[:,0]==1
    AmpTac=np.unique(Psicof[Tactile,1])
    AmpAud=np.unique(Psicof[Auditive,2])
    Ytac=np.zeros((np.size(AmpTac,0),1))
    Yaud=np.zeros((np.size(AmpAud,0),1))
    colorT={6:'yellow',8:'orange',10:'red',12:'blue',24:'purple'};
    colorA={0.006:'yellow',0.1:'orange',0.5:'red',1:'blue',2.5:'purple'};
    Theta=np.zeros((100,11))
    R=np.zeros((100,11))    #Rayleigh statistic
    r=np.zeros((100,11))    #Vector strenght    
#Begin plot for zeros
    f2, (p1, p2, p3) = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    f3, (ax4, ax5, ax6) = plt.subplots(1, 3, sharex=True, sharey=True)
    Pos=np.unique((Zero*Hits)*np.arange(1, rep+1))
    y=0.0
    begin=0
    for i in range(1, np.size(Pos)):
        cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[i]+1, electrode, unit)
        spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[i]-1, 6]
        tmp=( spykes>0.05 )*( spykes<0.55  )
        A,Theta[begin,0], R[begin,0], r[begin, 0]=vecstrength(spykes[tmp], T)
        begin+=1
    
    Pos=np.unique( (Zero* ~(Hits) ) * np.arange(1, rep+1) )
    for i in range(1, np.size(Pos)):
        cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[i]+1, electrode, unit)
        spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[i]-1, 6]
        tmp=(spykes>-1)*(spykes<1.5)    
        y+=0.1
        tmp=( spykes>0.05 )*( spykes<0.55  )
        A,Theta[begin,0], R[begin,0], r[begin, 0]=vecstrength(spykes[tmp], T)
        begin+=1
    tmp=Theta[:,0]!=0    
    p1.scatter(Theta[tmp,0], r[tmp,0], c ='yellow', marker = "s" )  
    ax4.hist(R[tmp,0], bins=20, alpha=1, edgecolor = 'black',  linewidth=1)
#Begin plot for tactile
    Pos=np.unique((Tactile*Hits)*np.arange(1, rep+1))
    y=0.0
    for i in range(0, np.size(AmpTac)):
        begin=0
        lab="Amplitude: μm%s"%(AmpTac[i])
        Pos=((Psicof[:,1]==AmpTac[i])*Hits)*np.arange(1, rep+1)
        if(np.max(Pos)>0):
            Pos=np.unique(Pos)
            for j in range(1, np.size(Pos)):
                cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[j]+1, electrode, unit)
                spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[j]-1, 6]
                tmp=(spykes>-1)*(spykes<1.5) 
                y+=0.1
        #In this part calculates the theta value
                tmp=( spykes>0.05 )*( spykes<0.55  )
                A,Theta[begin,i+1], R[begin,i+1], r[begin, i+1]=vecstrength(spykes[tmp], T)
                begin+=1
        Pos=((Psicof[:,1]==AmpTac[i])* ( ~Hits))*np.arange(1, rep+1)
        if(np.max(Pos)>0):
            Pos=np.unique(Pos)
            for j in range(1, np.size(Pos)):
                    cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[j]+1, electrode, unit)
                    spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[j]-1, 6]
                    tmp=(spykes>-1)*(spykes<1.5)    
#                    ax2.scatter(spykes[tmp], y*np.ones(np.sum(tmp)), s = 5, c ='r', marker = ".")    
                    y+=0.1
                    tmp=( spykes>0.05 )*( spykes<0.55  )
                    A,Theta[begin, i+1], R[begin, i+1], r[begin, i+1]=vecstrength(spykes[tmp], T)
                    begin+=1
                    tmp=Theta[:,i+1]!=0    
        tmp=Theta[:,i+1]!=0
        p2.scatter(Theta[tmp,i+1], r[tmp,i+1], c =colorT[AmpTac[i]], marker = "+" , label=lab)  
        ax5.hist(R[tmp,i+1], bins=20, alpha=1, edgecolor = 'black', color=colorT[AmpTac[i]], linewidth=1, label=lab)
        Ytac[i]=y;

       
#Begin plot for Auditive
    plt.figure(4)
    ax=plt.subplot(111, projection='polar')
    Pos=np.unique((Auditive*Hits)*np.arange(1, rep+1))
    y=0.0
    for i in range(0, np.size(AmpAud)):
        begin=0
        lab="Amplitude: db%s"%(AmpAud[i])
        Pos=((Psicof[:,2]==AmpAud[i])*Hits)*np.arange(0, rep)
        if(np.max(Pos)>0):
            Pos=np.unique(Pos)
            for j in range(1, np.size(Pos)):
                cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[j]+1, electrode, unit)
                spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[j]-1, 6]
                tmp=(spykes>-1)*(spykes<1.5)
                y+=0.1
        #In this part calculates the theta value
                tmp=( spykes>0.05 )*( spykes<0.55  )
                A,Theta[begin,6+i], R[begin,i+6], r[begin, i+6]=vecstrength(spykes[tmp], T)
                begin+=1
        Pos=((Psicof[:,2]==AmpAud[i])* ( ~Hits))*np.arange(1, rep+1)
        if(np.max(Pos)>0):
            Pos=np.unique(Pos)
            for j in range(1, np.size(Pos)):
                cname="%s/Text/%s/%s.mat_2_%d_%d_%d.csv"%(path, file, file, Pos[j]+1, electrode, unit)
                spykes=np.loadtxt(cname, dtype=float)-Tiempos[Pos[j]-1, 6]
                tmp=(spykes>-1)*(spykes<1.5)    
                y+=0.1
                tmp=( spykes>0.05 )*( spykes<0.55  )
                A, Theta[begin, i+6], R[begin, i+6], r[begin, i+6]=vecstrength(spykes[tmp], T)
                begin+=1
                plt.figure(4), ax.scatter(A, np.ones((np.size(A, 0)), 1), c=colorA[AmpAud[i]], marker = "s"  )
    tmp=Theta[:,i+6]!=0
    p3.scatter(Theta[tmp,i+6], r[tmp,i+6], c=colorA[AmpAud[i]], marker = "s" , label=lab)  
    ax6.hist(R[tmp,i+6], bins=20, alpha=1, edgecolor = 'black', color=colorA[AmpAud[i]], linewidth=1, label=lab)
    Yaud[i]=y;
    f2.legend(loc='best')
    f3.legend(loc='best')
    plt.yticks(Ytac, AmpTac)
    plt.yticks(Yaud, AmpAud)        
#__________________________________________________________________________________________________________________    
#__________________________________________________________________________________________________________________
#__________________________________________________________________________________________________________________    
    
def Psicofisica(cfile, monkey, delim="\t"):
    """ This function calculates the percentage of responses per classs given a mat file. To do that needs 2 parameters:
        *1 The namefile, as can be RR032152_002 for example, is a stringfile
        *2 monkey, indicates the number in the experiment: 32 or 33 for database.
        *3 delimi, indicates deimiter for psycophysics file
        This function returns an array whose elements are: Tactile, amplitude in cue, Acoustic amplitude in cue. 
        Tactile amplitude in main stimulus, acoustic amplitude in main stimuli, responses at first button, responses at second button
        responses at third button:
        AT1     AA1       AT2       AA2      RB1       RB2       RB3   
    """
    from numpy import loadtxt, zeros, unique,  copy, array
    from numpy import sum as nsum
    from os import name
    import Phd_ext as Pext
    whso=name    
    if whso=='nt':
        if monkey==32:
            cpath='D:\BaseDatosKarlitosNatsushiRR032\Text'
            cname="%s\%s\%s.mat_1.csv"%(cpath, cfile, cfile)
        else:
            cpath='D:\Database_RR033\Text'
            cname="%s\%s\%s_1.csv"%(cpath, cfile, cfile)
    else:
        if monkey==32:
            cpath="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text"
            cname="%s/%s/%s.mat_1.csv"%(cpath, cfile, cfile)
        else:
            cpath="/run/media/sparra/AENHA/Database_RR033/Text"    
            cname="%s/%s/%s_1.csv"%(cpath, cfile, cfile)
    try:
        Psicof=loadtxt(cname,  delimiter=delim, dtype=float)
    except:
        Psicof=loadtxt(cname,  delimiter=",", dtype=float)
        #Psicof=Pext.SLoad(cname, ',')
    print("Archivo cargado.")
    try:        
        Psicof=copy(Psicof[:, array([2, 3, 8, 10, 11, 13])])
    except:
        print("vale ...")
    Psicofisica=[]
    classes=unique(Psicof[:, [2, 3, 4, 5] ], axis=0, return_counts=False)
    Psicofisica=zeros((len(classes), 7), dtype=float)
    Psicofisica[:, 0:4]=copy(classes)
    for nclas in range(len(classes)):
        indexes=(Psicof[:, 2]==classes[nclas, 0])*(Psicof[:, 3]==classes[nclas, 1])*(Psicof[:, 4]==classes[nclas, 2])*(Psicof[:, 5]==classes[nclas, 3])
        for button in range(1, 4):
            partial=Psicof[indexes, 0]==button
            Psicofisica[nclas, button+3]=nsum(partial)/nsum(indexes)
    return len(Psicof), Psicofisica
          
   


def PSTH(cfile, electrode, fs, rs, unit, status, csaving, kernel, return0):
    
    """
    This function creates a set of plots of the PSTH by using a causal window with a 
    rectified half wave kernel. Until now, thiw function is perfect only for uncertainty sets 
    
    This function has four parameters:
    *1 cfile: the name of the file to analyze    example="RR032152_002";
    *2 electrode: The number of the electrode 
    *3 fs:  Frequency sampling for creating the vector of events
    *4 rs: Size of the windows in ms
    *5 unit=2;
    *6 status A label for neuron to indicate if its in protocol or not.
    *7 csaving= directory where the pictures will be sabed. example: "/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/RS/activity/Set B";
    *8 kernel, 1 indicates rectified Half wave, 2 indicates Gaussian.
    *9 return0, 0 indicates that not return the psth data and no otherwise
    
    """
    # -*- coding: UTF-8 -*-
    import numpy as np
    from pyexcel_ods import get_data
    import matplotlib.pyplot as plt
    from math import ceil
    from Phd_ext import FRcausalR as FRcausalR_cython
  #  from scipy import optimize
    from os import name
    if name=='nt':
        cpath="D:\BaseDatosKarlitosNatsushiRR032\Text_s"
    else:
        cpath="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/Text_s"
#Definición de variables
#--------------------------------------------------------------------------------------------
#   Input parameters for converting this to a function
#    cfile="RR032189_002";
#    electrode=5;
#    unit=1;
#    status="cuidada";
#    csaving="/run/media/sparra/AENHA/BaseDatosKarlitosNatsushiRR032/RS/activity/A1/Set B";
#    fs=2000;
#    kernel=3;
#    rs=100; #Resolution amplitude of windows.      
    #--------------------------------------------------------------------------------------------
    def test_func(x, a, b):
        return a*x+b
    
    form={6:'yellow',8:'orange',10:'red',12:'blue',24:'purple'};
    form2={0.006:'yellow',0.1:'orange',0.5:'red',1:'blue',2.5:'purple'};
    #Loading files
    cname="%s/%s/%s_Psyc.csv" %(cpath, cfile, cfile);
    print(cname)
    Psicof=np.loadtxt(cname, dtype=float); 
    cname="%s/%s/%s_T.csv" %(cpath, cfile, cfile);
    Tiempos=np.loadtxt(cname, dtype=float); 
    print(cname)
    cname="%s/%s/%s_e%d_u%d.csv" %(cpath, cfile, cfile, electrode, unit)
    print(cname)
    try:
        A=np.loadtxt(cname, usecols=(0),  delimiter=',')
    except:
        A=np.loadtxt(cname, usecols=(0),  delimiter='\t')
    A=A.astype(np.int32)-1
    Psicof=Psicof[A, :]
    Tiempos=Tiempos[A, :]
    blocelements=[0, 0, 0, 0, 0];
    bloque0=np.zeros(100);
    bloque1=np.zeros(100);
    bloque2=np.zeros(100);
    bloque3=np.zeros(100);
    bloque4=np.zeros(100);
    elements=np.zeros((25000,11));
    div=np.zeros(11);
    mean=sigma=0;
    Data=get_data(cname, delimiter="\t")
    Data=Data[cfile+'_e'+ str(electrode)+ '_u'+ str(unit)+'.csv']
    #print("Los trenes de espigas se cargaron como:, ", type(Data))
   # print(Data)
#Extracting the value to standardize data mu, sigma.
    for i in range (0,np.size(Tiempos,0)):
        data=Data[i][1::]        
        pd=data<=(Tiempos[i,0]);        
        mean+=np.sum(pd)/Tiempos[i,0];
        sigma+=(np.sum(pd)/Tiempos[i,0])**2;        
        del data, pd;    
    mean=mean/np.size(Tiempos,0);
    sigma=np.sqrt((sigma/np.size(Tiempos,0))-mean**2);
    #_------------------------------------------------------------------------------
# Conteo del número de elementos en cada bloque y #CREACIÓN DE LISTAS PARA CADA BLOQUE            

    for i in range (0,  np.size(Tiempos,0)):
        if(Psicof[i][8]==0 and Psicof[i][10]==0 and Psicof[i][11]==0 and Psicof[i][13]==0):
            bloque0[blocelements[0]]=i;
            blocelements[0]=blocelements[0]+1;       
        elif(Psicof[i][8]>=0 and Psicof[i][10]==0 and Psicof[i][11]>0 ):
            bloque1[blocelements[1]]=i;
            blocelements[1]=blocelements[1]+1;
        elif(Psicof[i][8]==0 and Psicof[i][10]>=0 and Psicof[i][13]>0):
            bloque2[blocelements[2]]=i;
            blocelements[2]=blocelements[2]+1;
        elif(Psicof[i][8]!=0 and Psicof[i][10]==0 and Psicof[i][11]==0 and Psicof[i][13]>=0):
            bloque3[blocelements[3]]=i;
            blocelements[3]=blocelements[3]+1;
        elif(Psicof[i][8]==0 and Psicof[i][10]>0 and Psicof[i][11]>=0 and Psicof[i][13]==0):
            bloque4[blocelements[4]]=i;
            blocelements[4]=blocelements[4]+1;
#_------------------------------------------------------------------------------         
        #Creating ones and zeros vector
#_------------------------------------------------------------------------------
    fr=np.ones((25000,blocelements[1]));
    fr=fr*-100;
    tiempos=np.zeros((25000,blocelements[1]));
    for i in range(0, blocelements[1]):
        data=np.array(Data[int(bloque1[i])][1::]);
        numel=ceil(data[-1]*fs);
#Starting with the calculation of firing rate.
        temporal=np.arange(0,numel/fs,1/fs);
        tiempos[0:np.size(temporal),i]=np.copy(temporal)-Tiempos[int(bloque1[i]),6];
        pd=int(Tiempos[int(bloque1[i]),0]*fs);
        pd2=int(Tiempos[int(bloque1[i]),6]*fs);
        if(kernel==1):
            FR2=FRcausalR_cython(data, rs/1000,fs); #100ms =0.1 sec
        elif (kernel==2):
            FR2=FRGauss(data,rs/1000,fs,1);
        else:
            FR2=FRGauss(data,rs/1000,fs,0);
        if(sigma>0):
            FR2=(FR2-mean)/sigma;   #Standardizing data    
        else:
            FR2=FR2-mean;
        fr[0:np.size(FR2,0),i]=FR2;
        #fig0, ax0=plt.subplots(1,1, figsize=(12, 9))
        #ax0.plot(np.linspace(-4,-2, 2*fs),fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],form[Psicof[int(bloque1[i]),11]]);
        #ax0.plot(tiempos[int(pd2-(1.75*fs)):np.size(temporal),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i],form[Psicof[int(bloque1[i]),11]]);
        #plt.show();
        size=np.size(temporal)-int(pd2)+int(1.75*fs)+int(2*fs);
        if(Psicof[int(bloque1[i]),11]==6):
            elements[0:size,0]=elements[0:size,0]+np.concatenate((fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[0]=div[0]+1;
        elif(Psicof[int(bloque1[i]),11]==8):
            elements[0:size,1]=elements[0:size,1]+np.concatenate((fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[1]=div[1]+1;
        elif(Psicof[int(bloque1[i]),11]==10):
            elements[0:size,2]=elements[0:size,2]+np.concatenate((fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[2]=div[2]+1;
        elif(Psicof[int(bloque1[i]),11]==12):
            elements[0:size,3]=elements[0:size,3]+np.concatenate((fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[3]=div[3]+1;
        else:
            elements[0:size,4]=elements[0:size,4]+np.concatenate((fr[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[4]=div[4]+1;
    #ax0.set_xlabel("Tiempo [s]");
    #ax0.set_ylabel("Tasa de disparo ");
    title="%s_%d_%d       %s"%(cfile,electrode,unit,status);
    #ax0.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    #ax0.axvspan(0, 0.5, alpha = 0.35) 
    #ax0.set_title(title);
    Title="%s/%s_%d_%dT.png"%(csaving,cfile,electrode,unit);
    #fig0.savefig(Title);
    tt=np.concatenate((np.linspace(-4,-2,(2*fs)),np.arange(-1.5-0.15,5-0.15,1/fs)))+rs/1000;
    elements[:,0]=elements[:,0]/div[0];
    elements[:,1]=elements[:,1]/div[1];
    elements[:,2]=elements[:,2]/div[2];
    elements[:,3]=elements[:,3]/div[3];
    elements[:,4]=elements[:,4]/div[4];
    fig1, ax1=plt.subplots(1,1, figsize=(12, 9))
    ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),0],'yellow',label='0 $\mu$m');
    ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),1],'orange',label='8 $\mu$m');
    ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),2],'red',label='10 $\mu$m');
    ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),3],'blue', label='12 $\mu$m');
    ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),4],'purple',label='24 $\mu$m');
    ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),0],'yellow');
    ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),1],'orange');
    ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),2],'red');
    ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),3],'blue');
    ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),4],'purple');
    ax1.set_xlabel("Time [s]", fontsize=28)
    ax1.set_ylabel("Neural activity", fontsize=28);
    #ax1.legend(loc='best', frameon=False, fontsize=14),     
    #ax1.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    ax1.axvspan(0,0.5, alpha = 0.35) 
    ax1.legend(frameon=False, fontsize=26)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis='x', labelsize=26 ) 
    ax1.tick_params(axis='y', labelsize=26 ) 
    ax1.axis([-4, 3.5, -2, 14])
    #ax.span "Top"set_visible(False)
#__________________________________________________________________________________________________________________________________
    fr2=np.ones((25000,blocelements[2]));
    fr2=fr2*-100;
    tiempos2=np.zeros((25000,blocelements[2]));
    for i in range(0, blocelements[2]):
        #cname="%s/%s/%s.mat_2_%d_%d_%d.csv" %(cpath, cfile, cfile,2+bloque2[i],electrode, unit);
        data=np.array(Data[int(bloque2[i])][1::]);
        numel=ceil(data[-1]*fs);
#Starting with the calculation of firing rate.
        temporal=np.arange(0,numel/fs,1/fs);
        tiempos2[0:np.size(temporal),i]=np.copy(temporal)-Tiempos[int(bloque2[i]),6];
        pd=int(Tiempos[int(bloque2[i]),0]*fs);
        pd2=int(Tiempos[int(bloque2[i]),6]*fs);
        if(kernel==1):
            FR2=FRcausalR_cython(data,rs/1000,fs); #100ms =0.1 sec
        elif (kernel==2):
            FR2=FRGauss(data,rs/1000,fs,1);
        else:
            FR2=FRGauss(data,rs/1000,fs,0);
        if(sigma>0.0001):
            FR2=(FR2-mean)/sigma;   #Standardizing data    
        else:
            FR2=FR2-mean;
        fr2[0:np.size(FR2,0),i]=np.copy(FR2);
        #fig3, ax3=plt.subplots(1,1, figsize=(12, 9))
        #ax3.plot(np.linspace(-4,-2,(2*fs)),fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],form2[Psicof[int(bloque2[i]),13]]);
        #ax3.plot(tiempos2[int(pd2-(1.75*fs)):np.size(temporal),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i],form2[Psicof[int(bloque2[i]),13]]);
        #plt.show();
        size=np.size(temporal)-int(pd2)+int(1.75*fs)+int(2*fs);
        if(Psicof[int(bloque2[i]),13]==0.006):
            elements[0:size,5]=elements[0:size,5]+np.concatenate((fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[5]=div[5]+1;
        elif(Psicof[int(bloque2[i]),13]==0.1):
            elements[0:size,6]=elements[0:size,6]+np.concatenate((fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[6]=div[6]+1;
        elif(Psicof[int(bloque2[i]),13]==0.5):
            elements[0:size,7]=elements[0:size,7]+np.concatenate((fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[7]=div[7]+1;
        elif(Psicof[int(bloque2[i]),13]==1):
            elements[0:size,8]=elements[0:size,8]+np.concatenate((fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[8]=div[8]+1;
        else:
            elements[0:size,9]=elements[0:size,9]+np.concatenate((fr2[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr2[int(pd2-(1.75*fs)):np.size(temporal),i]));
            div[9]=div[9]+1;
    #ax3.set_xlabel("Tiempo [s]", fontsize=14);
    #ax3.set_ylabel("Tasa de disparo", fontsize=14);
    title="%s_%d_%d       %s"%(cfile,electrode,unit,status);
    #ax3.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    #ax3.axvspan(0,0.5, alpha = 0.35) 
    #ax3.set_title(title);
    #Title="%s/%s_%d_%dA.png"%(csaving,cfile,electrode,unit);
    #fig3.savefig(Title);
    tt=np.concatenate((np.linspace(-4,-2,(2*fs)),np.arange(-1.5-0.15,5-0.15,1/fs)))+rs/1000;
    elements[:,5]=elements[:,5]/div[5];
    elements[:,6]=elements[:,6]/div[6];
    elements[:,7]=elements[:,7]/div[7];
    elements[:,8]=elements[:,8]/div[8];
    elements[:,9]=elements[:,9]/div[9];
    fig4, ax4=plt.subplots(1, 1, figsize=(12, 9))
    ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),5],'yellow',label='0 db');
    ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),6],'orange',label='35 db');
    ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),7],'red',label='45 db');
    ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),8],'blue', label='51 db');
    ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),9],'purple',label='60 db');
    ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),5],'yellow');
    ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),6],'orange');
    ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),7],'red');
    ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),8],'blue');
    ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),9],'purple');
    #ax4.legend(loc='best', frameon=False), 
    ax4.set_xlabel("Time [s]", fontsize=28), 
    ax4.set_ylabel("Neural activity", fontsize=28);
    #ax4.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    ax4.axvspan(0, 0.5, alpha =0.35) 
    #ax4.set_title(title);
    ax4.legend(frameon=False, fontsize=26)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.tick_params(axis='x', labelsize=26 ) 
    ax4.tick_params(axis='y', labelsize=26 ) 
    ax4.axis([-4, 3.5, -2, 14])
#_________________________________________________________________________________________________________________________

    fr3=np.ones((25000,blocelements[0]));
    fr3=fr3*-100;
    tiempos3=np.zeros((25000,blocelements[0]));
    for i in range(0, blocelements[0]):
        #cname="%s/%s/%s.mat_2_%d_%d_%d.csv" %(cpath, cfile, cfile,2+bloque0[i],electrode, unit);
        data=np.array(Data[int(bloque0[i])][1::]);
        numel=ceil(data[-1]*fs);
        temporal=np.arange(0,numel/fs,1/fs);
        tiempos3[0:np.size(temporal),i]=np.copy(temporal)-Tiempos[int(bloque0[i]),6];
        temporal=np.arange(0,numel/fs,1/fs);
        pd2=int(Tiempos[int(bloque0[i]),6]*fs);
        pd=Tiempos[int(bloque0[i]),0]*fs;
        pd2=ceil(pd2);
#Starting with the calculation of firing rate.
        if(kernel==1):
            FR2=FRcausalR_cython(data,rs/(0.5*fs),fs); #100ms =0.1 sec
        elif (kernel==2):
            FR2=FRGauss(data,rs/(0.5*fs),fs,1);
        else:
            FR2=FRGauss(data,rs/(0.5*fs),fs,0);
        if(sigma>0):
            FR2=(FR2-mean)/sigma;   #Standardizing data   
        else:
            FR2=FR2-mean;
        fr3[0:np.size(FR2,0),i]=np.copy(FR2);
        #fig6, ax6=plt.subplots(1,1, figsize=(12, 9))
        #ax6.plot(np.linspace(-4,-2,(2*fs)),fr3[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],'yellow');
        #ax6.plot(tiempos3[int(pd2-(1.75*fs)):np.size(temporal),i],fr3[int(pd2-(1.75*fs)):np.size(temporal),i],'yellow');
        #ax6.plot(np.linspace(-4,-2,(2*fs)),fr3[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],'yellow', label="0 $\mu$ m");
        #ax6.plot(tiempos3[int(pd2-(1.75*fs)):np.size(temporal),i],fr3[int(pd2-(1.75*fs)):np.size(temporal),i],'yellow');
        #ax6.plot(np.linspace(-4,-2,(2*fs)),fr3[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],'yellow', label="0 db");
        #ax1.plot(tiempos3[int(pd2-(1.75*fs)):np.size(temporal),i],fr3[int(pd2-(1.75*fs)):np.size(temporal),i],'yellow');

        #plt.show();
        size=np.size(temporal)-int(pd2)+int(1.75*fs)+int(2*fs);
        elements[0:size,10]=elements[0:size,10]+np.concatenate((fr3[int(pd-(0.5*fs)):int(pd+(1.5*fs)),i],fr3[int(pd2-(1.75*fs)):np.size(temporal),i]));
   # tt=np.concatenate((np.linspace(-4,-2,(2*fs)), np.arange(-1.5-0.15, 5-0.15,1/fs)))+rs/1000;
    elements[:,10]=elements[:,10]/i;
    #ax6.set_xlabel("Tiempo [s]", fontsize=14);
    #ax6.set_ylabel("Tasa de disparo", fontsize=14);
   # title="%s_%d_%d       %s"%(cfile,electrode,unit,status);
    #ax6.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    #ax6.axvspan(0,0.5, alpha = 0.35) 
    #Title="%s/%s_%d_%dZ.png"%(csaving,cfile,electrode,unit);
    #ax6.set_title(title);
    #fig6.savefig(Title);
    
    #fig8, ax8=plt.subplots(1,1, figsize=(12, 9))
    #ax8.plot(tt[0:(2*fs)],elements[0:(2*fs),10],'yellow',label='no estímulo');
    #ax8.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),10],'yellow');
   # plt.figure(8), plt.legend(loc='best', frameon=False)
    #ax8.set_xlabel("Tiempo [s]");
    #ax8.set_ylabel("Tasa de disparo ");
    
    #ax1.plot(tt[0:(2*fs)],elements[0:(2*fs),10],'yellow',label='0 $\mu$m');
    #ax1.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),10],'yellow');
    #plt.figure(1), plt.legend(loc='best', fontsize=14)
    
   # ax4.plot(tt[0:(2*fs)],elements[0:(2*fs),10],'yellow',label='0 db');
    #ax4.plot(tt[(2*fs):np.size(tt)],elements[(2*fs):np.size(tt),10],'yellow');
    #plt.figure(4), plt.legend(loc='best', fontsize=14)
    
   # title="%s_%d_%d       %s"%(cfile,electrode,unit,status);
    #ax4.axvspan(Tiempos[1,8]-Tiempos[1,6],Tiempos[1,8]-Tiempos[1,6]+0.2, alpha = 0.25) 
    #ax8.axvspan(0,0.5, alpha = 0.35) 
   # Title="%s/%s_%d_%dZb.png"%(csaving,cfile,electrode,unit);
    #Title2="%s/%s_%d_%dZb.svg"%(csaving,cfile,electrode,unit);
   # fig8.savefig(Title), fig8.savefig(Title2);    
    #Title="%s/%s_%d_%dAb.png"%(csaving,cfile,electrode,unit);
    #Title2="%s/%s_%d_%dAb.svg"%(csaving,cfile,electrode,unit);
   # fig4.savefig(Title), fig4.savefig(Title2);
    #plt.show()
    #Title="%s/%s_%d_%dTb.png"%(csaving,cfile,electrode,unit);
    #Title2="%s/%s_%d_%dTb.svg"%(csaving,cfile,electrode,unit);
    #fig1.savefig(Title), fig1.savefig(Title2);
    #plt.show()
#
    if(return0!=0):
        A=[tt, elements]
        return A
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def permutations(setA, setB, nperm, fun):
    """
    Function written by Sergio Parra Sánchez.
    
    This function caculates a permutation test, for that uses a two terms, this function
    needs two arrays with the same size    
    """
    from numpy import size, zeros, sqrt
    from random import randint
    sz=size(setA)
    A=[]
    set1me, set2me= 0, 0
    set1std, set2std= 0, 0
    if(size(setA)!=size(setB)):
        raise input('Size of sets is different')
    for i in range(0, nperm):
        for j in range(0, sz):
            if(randint(0, 1))==1:
              set1me+=setA[j]
              set2me+=setB[j]
              set1std+=(setA[j]**2)
              set2std+=(setB[j]**2)
            else:
              set1me+=setB[j]
              set2me+=setA[j]
              set1std+=(setB[j]**2)
              set2std+=(setA[j]**2) 
        sigma1=set1std/sz-(set1me/sz**2)
        sigma2=set2std/sz-(set2me/sz**2)
        A.append(((set1me-set2me)/sz )/sqrt(sigma1**2 + sigma2**2) )
    return A    
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def responsivity(cfile, electrode, unit, cpath, epoch2, Tanalize, wlength, step, Nperm):
    """
        This function calculates responsivity based on firing rate. This function requires the following 
        input parameters:
            * cfile:          File to analize responsivity
            * electrode       Electrode to analize 
            * unit            unit to analize
            * cpath           path where data is
            * epoch1          Beginning for basal window. Write according to T dictionary
            * epoch2          Beggining for the analisis window for resposivity
            * Taanalize       Indicates the length of the windows time for analysis
            * wlength         Length of window 
            * step            Step for moving window
            * Nperm           Number of permutations to get the significance level.
    """
    T={"PD":0, "KD":1, "iE1":2, "fE1":3, "iE2":4, "fE2":5, "iE3":6, "fE3":7, "PU":8, "KU":9, "PB":10}
    from pyexcel_ods import get_data
    from numpy import arange, loadtxt, copy, max, zeros, sum, unique
    import AUROC
    import matplotlib.pyplot as plt
    cname="%s\\%s\\%s_e%d_u%d.csv"%(cpath, cfile, cfile, electrode, unit)
    data=get_data(cname)
    spikes=data[('%s_e%d_u%d.csv'%(cfile, electrode, unit))] 
    A=loadtxt(cname, usecols=(0), dtype=int, delimiter=',')
    A-=1
    cname="%s\\%s\\%s_T.csv"%(cpath, cfile, cfile)
#    Tiempo=loadtxt(cname, usecols=(0, 1, 4, 5, 6, 7), dtype=float)
    Tiempo=loadtxt(cname, dtype=float)
    Tiempos=copy(Tiempo[A, :])
    cname="%s\\%s\\%s_Psyc.csv"%(cpath, cfile, cfile)
    Psic=loadtxt(cname, usecols=(8, 10, 11, 13), dtype=float)
    Psicof=copy(Psic[A, :])
    resp=zeros((500, 2), dtype=float)
    perm=[zeros((500, int(Nperm)), dtype=float), zeros((500, int(Nperm)), dtype=float) ]
    del Psic, Tiempo
    MA, MT= max(Psicof[:, 2]), max(Psicof[:, 3])
    if MA>0 or MT>0: #Determines that set belongs to uncertainty condition
        #First, calculate responsivity for tactile amplitude
        for j in [2, 3]:  #Loop that swaps per type of stimuli, Acoustic or tactile
            Ind=unique((Psicof[:, j]>0)*arange(1, len(Psicof)+1))[1::]
            Ind-=1
            fir1=zeros((len(Ind), 1), dtype=float)
            fir2=zeros((len(Ind), 1), dtype=float)
            pos=0 #Calculating base firing rate
            for i in Ind:
                ind=(spikes[i][ 1::]>=Tiempos[i, T[epoch2]]-wlength)*(spikes[i][1::]<=(Tiempos[i, T[epoch2] ]))
                fir1[pos]=sum(ind)
                pos+=1
            adv=0
            while(True):
                pos=0 #Calculating comparing firing rate
                for i in Ind: 
                    begin=Tiempos[i, T[epoch2]] + step*adv               
                    ind=(spikes[i][ 1::]>=begin)*(spikes[i][ 1::]<=(begin + wlength))
                    fir2[pos]=sum(ind)
                    pos+=1
                if j==2:            
                     resp[adv, j-2]=AUROC.AUROC(fir1, fir2, 1)  #Now begins AUROC calculation
                else:
                     resp[adv, j-2]=AUROC.AUROC(fir1, fir2, 1)  #Now begins AUROC calculation  
                perm[j-2][adv, :]=aurocperm(fir1, fir2, Nperm, AUROC.AUROC)                
                if (adv*step+wlength)<=Tanalize:
                    adv+=1
                else:
                    break
                
#               plt.figure(j, figsize=(9, 6))
#               plt.plot(Roc[:, 0], Roc[:, 1])        
    else:   #Determines that set belongs to distractors or focalization
       print('Nooooo') 
    time=arange(wlength, adv*step+wlength, step)
    plt.plot(time, resp[0:adv, 0], label='Tactile stimuli')
    plt.plot(time, resp[0:adv, 1], label='Acoustic stimuli')
    plt.legend(loc='best')
    return resp, perm
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def aurocperm(setA, setB, nperm, fun):
    """
    Function written by Sergio Parra Sánchez.
    
    This function caculates a permutation test, for that uses a two terms, this function
    needs two arrays with the same size, the number of permutations and the function of the parameter to evaluate of every
    permutation of sets.
    """
    from numpy import size, zeros, arange, concatenate, sort
    from random import shuffle, seed
    setfull=concatenate((setA, setB), axis=0)
    sz=size(setA)
    perm=zeros((nperm), dtype=float)
    if(size(setA)!=size(setB)):
        raise input('Size of sets is different')
    A=arange(0, sz*2)
    A.tolist()
    seed(0)
    for i in range(0, nperm):
        shuffle(A)
        perm[i]=fun(sort(setfull[A[0:sz]]), sort(setfull[A[sz::]]), 1)    #This only works when function has three parameters, the last being a number, 1 in this case.    
    return  perm
    
    
def Mutual_information(Pr, PrIs, Ps):
    """
    Program written by Sergio Parra Sánchez. 
    This function calculates the mutual information of the 2 distributions:
    *1 Pr    Probability of r
    *2 PrIs  Probability of r given s have ocurred
    *3 Ps    Probability of the stimulus
    
    
    This function calculates the mutual information according to the following
    expression:
        I=\sum_{r} \sum_{s} {  P(s)*P(r|s)*log2(   P(r|s)/(P(r) )   }
    
    Pr must be an array with two columns and multiple rows
    PrIs  must be dictionary where the elements are arrays containing the 
    distributions of P(r|s). Note that every array must be n by 2 where the 
    first row indicates r and the second the probability
    This first function needs that bins of distributions be equal for both
    distributions.
    
    """
    from numpy import log2
    Nstim=len(Ps)
    Nr=len(Pr)
    I=0 #Initialize the vlaue of mutual information
    for stim in range(Nstim): # sum over stimuli
        for r in range(Nr): #Sum over any neural-response-metric outcome
                 P=0
                 for stim2 in range(Nstim):
                     P+=PrIs[stim2][r]*Ps[stim2]  #Marginal probability
                 if P>0 and PrIs[stim][r]>0:# By convention zero develops zero
                     I+=Ps[stim]*PrIs[stim][r]*log2(PrIs[stim][r]/P)
    #This loops can be optimized by writing in Cython               
    return I
        



def psicofisica_archivos(file_database, cpath, monkey, save_filename):
    """
    Function created by Sergio Parra Sánchez
    
    This function creates a file (in the path function) with the proportion of responses for each possible answer per class.
    This function utilizes the information of database, the functionality of this function is based on the 
    assumption that in universal for all databases that can you get with the basis of a given structure. To access
    at such structure, please see the following files:
    D:\Drive_sparra\RS\S1_limpieza_RR033.xlsx
    D:\Drive_sparra\RS\S1_limpieza2.xlsx 
    
    This function needs of the following input parameters:
     * file_database, is the file name with path, as above. File must be an xls or xlsx file
     * cpath, indicates the general cpath where files (refered in database) will be found.
     * monkey, indicates the number of the monkey
     * save_filename, indicates the name of file created with the information of psychophysics
    
     
    """ 
    from pyexcel_xls import get_data
    from Phd import Psicofisica as Psicofisica
    from numpy import zeros, unique, copy, savetxt, size
    Database=get_data(file_database)
    Database=Database["Sheet1"] 
    Psict=zeros((40000, 11), dtype=float)
    headers=(Database[0])
    set_index=[]
    for header in range(5, len(headers)):
        if headers[header][0:3]=="set" or headers[header][0:3]=="Set":
            set_index.append(header)
    Nrows=len(Database)
    psicofi=0
    psicofe=0
    rows=1
    while(rows<(Nrows-1)):
        if len(Database[rows])==0 or type(Database[rows][0])==str:
            rows+=1
            continue
        serie=Database[rows][0]
        ini=rows
        fin=ini+1
        while(True):            
            if(fin<(Nrows-1) and len(Database[fin])>0):
                fin+=1
            else:
                break
        segment=zeros((fin-ini, len(set_index)), dtype=int)
        for segment_part in range(ini, fin):
            for cell in range (set_index[0], set_index[-1]+1):
                index_i, index_j=segment_part-ini, cell-set_index[0]
                if type(Database[segment_part][cell]) != str:
                    celda=float(Database[segment_part][cell])
                    if ( celda%1 )==0:
                        segment[index_i, index_j]=celda
       #Aquí empezará la intromisión de elementos de la serie a la psicofísica
        serie_elem=unique(segment)
        dic_sets=zeros((12, 1), dtype=int)
        for identifier in range(size(segment, 1)):
            unqelem=unique(segment[:, identifier])
            if unqelem[0]==0 and len(unqelem)>1:
                unqelem=copy(unqelem[1::])
            for ident in unqelem:
                dic_sets[ident-1]=identifier+1
                    
        if serie_elem[0]==0:
           serie_elem=copy(serie_elem[1::])
        for sets in serie_elem:
           if sets<10 and serie>=100:
               cfile="RR0%d%d_00%d"%(monkey, serie, sets)
           elif sets<10 and serie<10:
               cfile="RR0%d00%d_00%d"%(monkey, serie, sets)
           elif sets<10 and (serie>=10 and serie<100):
               cfile="RR0%d0%d_00%d"%(monkey, serie, sets)
           elif sets>=10 and serie>=100:
               cfile="RR0%d%d_0%d"%(monkey, serie, sets)
           elif sets>=10 and serie<10:
               cfile="RR0%d00%d_0%d"%(monkey, serie, sets)               
           else:
               cfile="RR0%d0%d_0%d"%(monkey, serie, sets)
           psicofisica=Psicofisica(cfile, monkey)
           psicofe=psicofi + len(psicofisica[1])
           Psict[psicofi:psicofe, 0]=monkey
           Psict[psicofi:psicofe, 1]=serie
           Psict[psicofi:psicofe, 2]=sets
           Psict[psicofi:psicofe, 3]=dic_sets[sets-1]
           Psict[psicofi:psicofe, 4::]=psicofisica[1]
           psicofi=psicofe
        rows=fin
        ini=fin
    savetxt(save_filename, Psict[0:psicofe, :], delimiter=',')
    return Psict[0:psicofe, :]  


#def dprime(psicofisica):
#    """
#    Función escrita por Sergio Parra Sánchez. 
#    Esta función en conjunto con la función Psicofisica permiten calcular el valor
#    de la d prima mediante la separación de la tarea en dos detecciones simples.
#    Para ello único argumento de entrada debe ser un arreglo matricial con el número
#    de renglones igual al número de clases. El arreglo debe tener la misma estructura que la 
#    devuelta por la función Psicofisica, por favor, vea esa documentación.
#    
#    
#    Esta función devuelve un arreglo con dos elementos, el primero es la d prima de la
#    detección táctil y el segundo la d prima en la detección acústica.
#
#
#    """
#    if sum(psicofisica[:, 0])==0: #Set de incertidumbre
#        find=(psicofisica[:, 2]>=0)*(psicofisica[:, 3]==0)
#        
#        
#    else:   #Set con cue
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ ----------------------------------   
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------         
    
def fir_Inf_pfile(monkey, serie, whset, met, guardar, graficar, periodo):
    """
    Esta función calculará el firing rate por conteo en una época definida así como la
    sincronización a 20Hz y las métricas de información como en el archivo Prueba_informacion_serie_RR032164.py.
    Esta función buscará los archivos en la base de datos ya establecida en el disco duro para los dos monos:
    RR032 y RR033 de modo que ya no será necesario indicar tal parámetro. Asímismo buscará en los archivos resumen
    la información necesaria sobre el set, y demás. Tales archivos son:
        S1_limpieza2.xlsx para el mono RR032 en el área S1 evidentemente.
        S1_limpieza_RR033.xlsx para el mono RR033 en el área S1.
        
        
        
    La función requiere de los siguientes argumentos de entrada:
        mono
        serie
        whset
        met:      ¿Cuál métrica? 0 indica firing rate, 1 indica sincronización, 2 periodicidad, otro indica coeficiente de variación local
        guardar
        graficar
        periodo: Una lista de cadena de carácteres con 4 o 6 elementos:
              *   Si son 4 solo indica los primeros 4, si son 6, el set completo.
                - el primero indica el tiempo inicial  (en términos de la tarea  de acuerdo a la lista T en Phd)               
                - el segundo indica el tiempo final en términos de la tarea, de manera similar a el formato del primer elemento
                - el tercero indica el desfase del inicio, negativo hacia atrás, positivo adelante (en s)
                - el cuarto indica el desfase del final, negativo atrás, positivo adelante (en s)
                - el quinto indica el ancho de la ventana (en s)
                - el sexto indica el tamaño del paso (en s)
        
        
        
    Ejemplo de su uso:  
    """
    from os import name
    from scipy import fftpack
    from Phd import T
    import Phd_ext as Pext
    from pyexcel_ods import get_data as getcsv
    from pyexcel_xls import get_data 
#    if graficar==1:
#        import matplotlib.ticker as ticker
#        import matplotlib
#        import matplotlib.pyplot as plt
#        matplotlib.rcParams.update({'font.size': 14, 'font.weight':'bold', 'font.style':'normal'})
    from numpy import loadtxt, array, copy, zeros, savetxt, size
    if name=='nt':
        if monkey==32:
            cpath="D:\BaseDatosKarlitosNatsushiRR032/Text_s"
            cfilex="D:\Drive_sparra\RS\S1_limpieza2.xlsx"
        elif monkey==33:
            cpath="D:\Database_RR033\Text_s"
            cfilex="D:\Drive_sparra\RS\S1_limpieza_RR033.xlsx "
        else:
            print("Ese mono no está contemplado \n")
            return 1
    else:
        if monkey==32:
            cpath="run\media\sparra\AENHA\BaseDatosKarlitosNatsushiRR032/Text_s"
        elif monkey==33:
            cpath="run\media\sparra\Trabajo\Database_RR033\Text_s"
        else:
            print("Ese mono no está contemplado \n")
            return 1
    Database=get_data(cfilex)
    Database=Database['Sheet1']
    headers=(Database[0])
    set_index=[]  #Elements to analize
    if whset=='all' or whset=="ALL":
        for header in range(5, len(headers)):
            if headers[header][0:3]=="set" or headers[header][0:3]=="Set":
                set_index.append(header)
    else:
        for sets in whset:
            for header in range(5, len(headers)):
                if headers[header]==sets:
                    set_index.append(header)        
    Nrows=len(Database)
    rows=1
    set_index=array(set_index)
    if guardar==1:
        if len(periodo)==4:
            if met==0:
                savefile="32_%d_full_%s%d_%s%d_firing.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3])
                file=open(savefile, 'a')
            elif met==1:
                savefile="32_%d_full_%s%d_%s%d_sync.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3])
                file=open(savefile, 'a')
            else:
                savefile="32_%d_full_%s%d_%s%d_CV2.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3])
                file=open(savefile, 'a')
        else:
            if met==0:
                savefile="32_%d_ventaneo_%s%d_%s%d_w%f_step%f_firing.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3], periodo[4], periodo[5])
                file=open(savefile, 'a')
            elif met==1:
                savefile="32_%d_ventaneo_%s%d_%s%d_w%f_step%f_VS.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3], periodo[4], periodo[5])
                file=open(savefile, 'a')
                savefile="32_%d_ventaneo_%s%d_%s%d_w%f_step%f_Ray.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3], periodo[4], periodo[5])
                file2=open(savefile, 'a')
            else:
                savefile="32_%d_ventaneo_%s%d_%s%d_w%f_step%f_CV2.csv"%(serie, periodo[0], periodo[2], periodo[1], periodo[3], periodo[4], periodo[5])
                file=open(savefile, 'a') 
    while(rows<(Nrows-1)):
        if len(Database[rows])==0 or Database[rows][0]!=serie:
            rows+=1
            continue
       #Aquí empezará la intromisión de elementos de la serie a la psicofísica       
        for sets in set_index:
           if Database[rows][sets]>0 and Database[rows][sets]%1==0:
                if Database[rows][sets]<10 and serie>=100:
                    cfile="RR0%d%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]<10 and serie<10:
                    cfile="RR0%d00%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]<10 and (serie>=10 and serie<100):
                    cfile="RR0%d0%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]>=10 and serie>=100:
                    cfile="RR0%d%d_0%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]>=10 and serie<10:
                    cfile="RR0%d00%d_0%d"%(monkey, serie, Database[rows][sets])               
                else:
                    cfile="RR0%d0%d_0%d"%(monkey, serie, Database[rows][sets])
                cfile2="%s\%s\%s_e%d_u%d.csv"%(cpath, cfile, cfile, Database[rows][2], Database[rows][3] )
           else:
               continue   
           try:
               datafile=Pext.SLoad(cfile2, ',')
           except:
               datafile=Pext.SLoad(cfile2, '\t') 
           if len(datafile[3])==0:               
               datafile=Pext.SLoad(cfile2, '\t') 
           if len(datafile[3])==0: 
               print("No fue posible abrir el archivo ", cfile2)
               file.close()
               if met==1:
                   file2.close()
               return 1
           try:
               A=loadtxt(cfile2, usecols=(0), dtype=int, delimiter=',')
           except:
               try:
                   A=loadtxt(cfile2, usecols=(0), dtype=int, delimiter='\t')
               except:
                   print("Posiblemente el renglón está lleno de ceros \n")
                   return cfile2
           A=A-1          
           cfile2="%s\%s\%s_Psyc.csv"%(cpath, cfile, cfile)
           try:
               Psyc_data=loadtxt(cfile2, dtype=float, delimiter='\t')
           except:
               Psyc_data=loadtxt(cfile2, dtype=float, delimiter=',')
           Psyc_data=copy(Psyc_data[A, :])
           cfile2="%s\%s\%s_T.csv"%(cpath, cfile, cfile)
           if len(periodo)==4:
               try:
                   Tiempos=loadtxt(cfile2, usecols=(0, T[periodo[0]], T[periodo[1]]), dtype=float, delimiter=',')
               except:
                   Tiempos=loadtxt(cfile2, usecols=(0, T[periodo[0]], T[periodo[1]]), dtype=float, delimiter='\t')
               Tiempos=copy(Tiempos[A, :])
               Tiempos[:, 0]=Tiempos[:, 0]
               Tiempos[:, 1]+=periodo[2]
               Tiempos[:, 2]+=periodo[3]
               if met==0:
                   metrica=Pext.firing_file(datafile, Tiempos, len(Tiempos), 1)
                   #metrica=firing_file(datafile, Tiempos, len(Tiempos), 1)
                   Psict=zeros((len(metrica), 13), dtype=float)
               elif met==1:
                   metrica=Pext.sync_file(datafile, Tiempos[:, 1: 2], len(Tiempos), 1/18.4)
                   Psict=zeros((len(metrica), 14), dtype=float)      
               elif met==2:
                   FR=Pext.firing_file_FFT(datafile, Tiempos, len(Tiempos), 0.050, 0.010 )
                   sample_freq = fftpack.fftfreq(85, 0.010) #Generación de frec. de muestreo
                   for i in range(len(Tiempos)):                      
                       sig_fft = fftpack.fft(FR)  
                       
                   return FR
               else:
#                   metrica=Localvar_file(datafile, Tiempos, len(Tiempos))
                   print("El coeficiente de variación local en ventanas muy restringidas no siempre puede ser calculado")
                   file.close()
                   return 1
               #Esta parte se debe llenar por ensayo
               for trial in range (len(metrica)):
                   Psict[trial, 0]=serie
                   Psict[trial, 1]=Database[rows][2]
                   Psict[trial, 2]=Database[rows][3]
                   Psict[trial, 3]=Database[rows][-1]
                   Psict[trial, 4]=Database[rows][-5]
                   Psict[trial, 5]=Database[rows][sets]
                   Psict[trial, 6]=sets-4
                   Psict[trial, 7]=Psyc_data[trial, 8]
                   Psict[trial, 8]=Psyc_data[trial, 10]
                   Psict[trial, 9]=Psyc_data[trial, 11]
                   Psict[trial, 10]=Psyc_data[trial, 13]
                   Psict[trial, 11]=Psyc_data[trial, 2]
                   Psict[trial, 12::]=metrica[trial, :]
               if(guardar==1):
                   savetxt(file, Psict, delimiter=',', newline='\n')
    
           elif len(periodo)==6 and guardar==1: #Cálculo de metrica por ventana
               Tiempos=loadtxt(cfile2, usecols=(T[periodo[0]], T[periodo[1]]), dtype=float)
               Tiempos=copy(Tiempos[A, :])
               Tiempos[:, 0]+=periodo[2]
               Tiempos[:, 1]+=periodo[3]
               Psict=zeros((len(datafile), 12), dtype=float) 
               for trial in range (len(datafile)):
                   Psict[trial, 0]=serie
                   Psict[trial, 1]=Database[rows][2]
                   Psict[trial, 2]=Database[rows][3]
                   Psict[trial, 3]=Database[rows][-1]
                   Psict[trial, 4]=Database[rows][-5]
                   Psict[trial, 5]=Database[rows][sets]
                   Psict[trial, 6]=sets-4
                   Psict[trial, 7]=Psyc_data[trial, 8]
                   Psict[trial, 8]=Psyc_data[trial, 10]
                   Psict[trial, 9]=Psyc_data[trial, 11]
                   Psict[trial, 10]=Psyc_data[trial, 13]
                   Psict[trial, 11]=Psyc_data[trial, 2]
               if met==0:
                   Pext.firing_file_wind(datafile, Tiempos, nrow=int(size(Tiempos, axis=0)), window=periodo[4], step=periodo[5], informationfile=Psict, file=file)
               elif met==1:
                   Pext.sync_file_wind(datafile, Tiempos, nrow=int(size(Tiempos, axis=0)), Period=1/18.4, window=periodo[4], step=periodo[5], informationfile=Psict, file1=file, file2=file2)   
               else:
                   Localvar_file_wind(datafile, Tiempos, nrow=int(size(Tiempos, axis=0)), window=periodo[4], step=periodo[5], informationfile=Psict, file=file)   
           else:
               print("Aún no se ha escrito, saludos ")
               return 1
        rows+=1 
    if guardar==1 and met==0:        
        file.close()
    elif met==1 and guardar==1 and len(periodo)>4:
        file.close()
        file2.close()
    elif guardar==1:
        file.close()
    return 0


def firing_file(data, epoch,  Nrow,  zt):
    """
    Dada una lista de arreglos como se suelen cargar los archivos de espigas y el inicio y
    final de un intervalo, esta función devuelve el número de cuentas que están en ese intervalo para
    cada elemento de la lista.
    
    zt es un indicador sobre la estandarización de los datos, si zt es igual a 1 se realizará la 
    transformada z dado el periodo anterior a la bajada de la punta.
    """
    from numpy import zeros, mean, std
    media=0
    sigma=1
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

def responsividad_Informacion(sfile, elec, unit, met, wn, stp, TrT, TrA):
    """
    Esta función calcula la responsividad de respuesta de una neurona utilizando las métricas
    de tasa, vector de fuerza, periodicidad por Fourier, etc. Por el momento solamente utiliza
    la métrica de tasa de disparo pero en el futuro se buscará que lo haga utilizando diversas métricas.
    Esta función tiene una fuerte cantidad de código en el archivo Phd_ext.pyx. Esto ha sido realizado así
    con el fin de optimizar la ejecución del código.
    
    Esta función calcula la tasa de disparo por ventanas de longitud wn y pasos stp para el cálculo de la 
    tasa de disparo por ensayo. Las tasas durante el periodo de estimulación son agrupadas en dos:
    zero-táctil (supraumbral TRT) y zero-sonoro (supraumbral TrA). Con esto se busca determinar la cantidad 
    de información dados dos estados (Presencia o no del estímulo) así como la significancia. Dada una información
    mutua relevante y significativa, entonces se podrá decir si la unidad es responsiva para los estímulos
    táctiles y/o auditivos. 
    
    Esta función requiere de los siguientes parámetros de entrada:
    * sfile: Es el nombre de la unidad, por ejemplo sfile="RR032164_002"
    * elec: Indica al electrodo con el cual se realizó la medición. elec=5
    * unit: Indica el número de unidad. unit=1
    * met:  Indica la metrica a utilizarse, hasta el momento soportada solo met=0: tasa de disparo
    * wn:   Indica la longitud de la ventana (en segundos) a utilizarse durante el periodo de estimulación.
    * stp:  Indica el paso con el que se moverá la ventana en segundos
    * TrT:  Indica el umbral táctil para el uso de las amplitudes en el análisis 9.8
    * TrA:  Indica el umbral sonoro para el uso de las amplitudes en el análisis 33.8
    sfile, elec, unit, met, wn, stp, TrT, TrA ="RR032164_002", 5, 1, 1, 0.05, 0.010, 9.8, 33.8
    
    """
    from pyexcel_ods import get_data
    from numpy import loadtxt, concatenate, array, random, arange, unique, reshape, histogram, zeros
    from numpy import sum as nsum
    from Phd_ext import firing as Pextfir
    from Information import Mutual_information
    spath="D:\\BaseDatosKarlitosNatsushiRR032\\Text_s\\"
    sname=spath + sfile +"\\"+ sfile + "_e" + str(elec)+ "_u" +str(unit)+ ".csv"
    data=get_data(sname, delimiter=',')
    data=data[sfile + "_e" + str(elec)+ "_u" +str(unit)+ ".csv"]
    if type(data[0][0])==str:
         data=get_data(sname, delimiter='\t')
         data=data[sfile + "_e" + str(elec)+ "_u" +str(unit)+ ".csv"]
    if type(data[0][0])==str:
         print("Archivo no leído correctamente \n")
         return -1
    data=array(data)
    for i in range(len(data)):
        data[i]=array(data[i])
    try:
        A=loadtxt(sname, usecols=(0), delimiter='\t', dtype=int)
    except:
        A=loadtxt(sname, usecols=(0), delimiter=',', dtype=int)
    A=A-1
    sname=spath + sfile +"\\"+ sfile + "_T.csv"
    try:
        Tiempos=loadtxt(sname, usecols=(0, 6, 7), delimiter='\t')
    except:
        Tiempos=loadtxt(sname, usecols=(0, 6, 7), delimiter=',')
    Tiempos=Tiempos[A, :]
    sname=spath + sfile +"\\"+ sfile + "_Psyc.csv"
    try:
        Psyc=loadtxt(sname, delimiter='\t', usecols=(8, 10, 11, 13))
    except:
        Psyc=loadtxt(sname, delimiter=',', usecols=(8, 10, 11, 13))
    Psyc=Psyc[A, :]
    Infdata=concatenate((Psyc, Tiempos), axis=1)
    del A, Psyc, Tiempos
    Infdata[:, 3]=fitmatlab(Infdata[:, 3])
    Maskzero=(Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]==0)*(Infdata[:, 3]==29.959941713970323)
    MaskTs=(Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]>TrT)*(Infdata[:, 3]==29.959941713970323)  # Captura los ensayos con amplitud T supra
    MaskAs=(Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]==0)*(Infdata[:, 3]>TrA)  # Captura los ensayos con amplitud A supra
    if met==1: #métrica: Tasa de disparo
        firZ=Pextfir(data[Maskzero], epoch=Infdata[Maskzero, 4::], Nrow=nsum(Maskzero),  zt=0, wn=wn, stp=stp) #Calcula la tasa
        firT=Pextfir(data[MaskTs], epoch=Infdata[MaskTs, 4::], Nrow=nsum(MaskTs),  zt=0, wn=wn, stp=stp)
        firA=Pextfir(data[MaskAs], epoch=Infdata[MaskAs, 4::], Nrow=nsum(MaskAs),  zt=0, wn=wn, stp=stp)    
        maximum=max([max(firT), max(firA), max(firT)])
        bins=arange(-1, maximum+1)
        PrIZ, bins=histogram(firZ, bins=bins, density=True )
        PrIT, bins=histogram(firT, bins=bins, density=True )
        PrIA, bins=histogram(firA, bins=bins, density=True )
        # Comienza el cálculo de la información
        PrIZ=reshape(PrIZ, (len(PrIZ), 1))
        PrIT=reshape(PrIT, (len(PrIT), 1))
        PrIA=reshape(PrIA, (len(PrIA), 1))        
        PrIs=concatenate((PrIZ, PrIT), axis=1)
        Ps=array([nsum(Maskzero)/(nsum(Maskzero)+ nsum(MaskTs)), nsum(MaskTs)/(nsum(Maskzero)+ nsum(MaskTs)) ])
        IRT=Mutual_information( PrIs, len(PrIZ), 2,  Ps)
        PrIs=concatenate((PrIZ, PrIA), axis=1)
        Ps=array([nsum(Maskzero)/(nsum(Maskzero)+ nsum(MaskAs)), nsum(MaskAs)/(nsum(Maskzero)+ nsum(MaskAs)) ])
        IRA=Mutual_information(PrIs, len(PrIZ), 2, Ps)
    else:
        print("Upsss... \n")
        return -1
    # comienza el cálculo de las permutaciones para evaluar la significancia de codificación.
    nperm=1000
    IRA_p=zeros((1, nperm), dtype=float)
    IRT_p=zeros((1, nperm), dtype=float)
    for perm in range(nperm):
        random.shuffle(Infdata)
        Maskzero=(Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]==0)*(Infdata[:, 3]==29.959941713970323)
        MaskTs = (Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]>TrT)*(Infdata[:, 3]==29.959941713970323)  # Captura los ensayos con amplitud T supra
        MaskAs = (Infdata[:, 0]==0)*(Infdata[:, 1]==0)*(Infdata[:, 2]==0)*(Infdata[:, 3]>TrA)  # Captura los ensayos con amplitud A supra        
        if met==1: #métrica: Tasa de disparo
            firZ=Pextfir(data[Maskzero], epoch=Infdata[Maskzero, 4::], Nrow=nsum(Maskzero),  zt=0, wn=wn, stp=stp) #Calcula la tasa
            firT=Pextfir(data[MaskTs], epoch=Infdata[MaskTs, 4::], Nrow=nsum(MaskTs),  zt=0, wn=wn, stp=stp)
            firA=Pextfir(data[MaskAs], epoch=Infdata[MaskAs, 4::], Nrow=nsum(MaskAs),  zt=0, wn=wn, stp=stp)    
            maximum=max([max(firT), max(firA), max(firT)])
            bins=arange(-1, maximum+1)
            PrIZ, bins=histogram(firZ, bins=bins, density=True )
            PrIT, bins=histogram(firT, bins=bins, density=True )
            PrIA, bins=histogram(firA, bins=bins, density=True )
            # Comienza el cálculo de la información
            PrIZ=reshape(PrIZ, (len(PrIZ), 1))
            PrIT=reshape(PrIT, (len(PrIT), 1))
            PrIA=reshape(PrIA, (len(PrIA), 1))        
            PrIs=concatenate((PrIZ, PrIT), axis=1)
            Ps=array([nsum(Maskzero)/(nsum(Maskzero)+ nsum(MaskTs)), nsum(MaskTs)/(nsum(Maskzero)+ nsum(MaskTs)) ])
            IRT_p[0, perm]=Mutual_information( PrIs, len(PrIZ), 2,  Ps)
            PrIs=concatenate((PrIZ, PrIA), axis=1)
            Ps=array([nsum(Maskzero)/(nsum(Maskzero)+ nsum(MaskAs)), nsum(MaskAs)/(nsum(Maskzero)+ nsum(MaskAs)) ])
            IRA_p[0, perm]=Mutual_information(PrIs, len(PrIZ), 2, Ps)
        else:
            print("Upsss... \n") #Aquí aún no llega ya que el anterior regresa el control al usuario 
    print("IRT=%f"%(IRT), "\t Significancia: %f \n"%(nsum(IRT_p<IRT)/nperm) )
    print("IRA=%f"%(IRA), "\t Significancia: %f \n"%(nsum(IRA_p<IRA)/nperm) )
    return 0

#_______________________________________________________________________________________________________________________
def Escribe_neurometria(monkey, serie, whset, guardar, graficar):
    """
    Esta función escribe un archivo de neurometría que irá creciendo conforme se incrementen los archivos limpios del 
    set B (Incertidumbre, que para el mono 32 generalmente son los 002).
    
    La neurometría será calculada mediante el uso del firing rate. Esta función utiliza la función optimumC de este mismo
    módulo. 
    Para hacer los cálculos de neurometría esta función buscará los archivos en la base de datos ya establecida en el disco duro para los dos monos:
    RR032 y RR033 de modo que ya no será necesario indicar tal parámetro. Asímismo buscará en los archivos resumen
    la información necesaria sobre el set, y demás. Tales archivos son:
        S1_limpieza2.xlsx para el mono RR032 en el área S1 evidentemente.
        S1_limpieza_RR033.xlsx para el mono RR033 en el área S1.        
        
    La función requiere de los siguientes argumentos de entrada:
        mono
        serie
        whset
        guardar
        graficar       
    """
    from os import name
    from Phd import optimumC
    from pyexcel_xls import get_data 
    if graficar==1:
        import matplotlib.ticker as ticker
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({'font.size': 14, 'font.weight':'bold', 'font.style':'normal'})
    from numpy import array, zeros, savetxt, concatenate
    Psict=zeros((1, 7), dtype=float)
    if name=='nt':
        if monkey==32:
            cpath="D:\BaseDatosKarlitosNatsushiRR032/Text_s"
            cfilex="D:\Drive_sparra\RS\S1_limpieza2.xlsx"
        elif monkey==33:
            cpath="D:\Database_RR033\Text_s"
            cfilex="D:\Drive_sparra\RS\S1_limpieza_RR033.xlsx "
        else:
            print("Ese mono no está contemplado \n")
            return 1
    else:
        if monkey==32:
            cpath="run\media\sparra\AENHA\BaseDatosKarlitosNatsushiRR032/Text_s"
        elif monkey==33:
            cpath="run\media\sparra\AENHA\Database_RR033\Text_s"
        else:
            print("Ese mono no está contemplado \n")
            return 1
    Database=get_data(cfilex)
    Database=Database['Sheet1']
    headers=(Database[0])
    set_index=[]  #Elements to analize
    if whset=='all' or whset=="ALL":
        for header in range(5, len(headers)):
            if headers[header][0:3]=="set" or headers[header][0:3]=="Set":
                set_index.append(header)
    else:
        for sets in whset:
            for header in range(5, len(headers)):
                if headers[header]==sets:
                    set_index.append(header)        
    Nrows=len(Database)
    rows=1
    set_index=array(set_index)
    if guardar==1:
        savefile="RR0%d_Unc_neurometria.csv"%(monkey)
        file=open(savefile, 'a')     
    while(rows<(Nrows-1)):
        if len(Database[rows])==0 or Database[rows][0]!=serie:
            rows+=1
            continue
       #Aquí empezará la intromisión de elementos de la serie a la psicofísica       
        for sets in set_index:
           if Database[rows][sets]>0 and Database[rows][sets]%1==0:
                if Database[rows][sets]<10 and serie>=100:
                    cfile="RR0%d%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]<10 and serie<10:
                    cfile="RR0%d00%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]<10 and (serie>=10 and serie<100):
                    cfile="RR0%d0%d_00%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]>=10 and serie>=100:
                    cfile="RR0%d%d_0%d"%(monkey, serie, Database[rows][sets])
                elif Database[rows][sets]>=10 and serie<10:
                    cfile="RR0%d00%d_0%d"%(monkey, serie, Database[rows][sets])               
                else:
                    cfile="RR0%d0%d_0%d"%(monkey, serie, Database[rows][sets])
           else:
               continue   
           Psict[0, 0]=serie
           Psict[0, 1]=Database[rows][2]      #Electrode
           Psict[0, 2]=Database[rows][3]      #Unit
           Psict[0, 3]=Database[rows][-1]     #3b?
           Psict[0, 4]=Database[rows][-5]     #CR
           Psict[0, 5]=Database[rows][sets]   #Set Num
           Psict[0, 6]=sets-4                 #Set Id 
           N=optimumC(cfile, Database[rows][2] , Database[rows][3])                              
           if(guardar==1):
              savetxt(file, concatenate((Psict, N[0], N[1]), axis=1), delimiter=',', newline='\n')            
           if(graficar==1):
               plt.figure(), plt.plot(N[0], c='blue'), plt.plot(N[1], c='red')
               plt.show()
        rows+=1 
    if guardar==1:   
        file.close()
    return 0



#----------------------- Las funciones a continuación van a la compilación en tiempo Jit
# En general estas funciones tienen una versión en cython, a la cual se podrá acceder desde
# la libreria Phd_ext. Salvo muy pocas excepciones verdaderamente especiales es que eso no se logrará
    
from numba import jit
from numpy import array,  zeros, size, arange, ceil, exp, pi, sqrt, savetxt, ediff1d, mean, isnan, std
from numpy import sum as nsum

@jit()
def FRcausalR(spiketimes, resolution, fs):
    """
    Author: Sergio Parra Sánchez
    This function creates the estimate in time of a firing rate dependent of time by
    using a causal half-wave rectification function as a kernel. 
    
    resolution must be in seconds.
    fs frequency sampling
    spiketimes is a vector with the event times (positive values) of each action potential
    
    For further information see Peter & Dayan, Theoretical Neuroscience, chapter 1
    
    """

    #from math imp
    alpha=1/resolution;
    numel=ceil(spiketimes[-1]*fs);
    t=arange(0,numel/fs,1/fs);
    for i in range(0,size(spiketimes)):
        FRa=alpha**2*(-t+spiketimes[i])*exp(-1*alpha*(-t+spiketimes[i]));
        for j in range(0,size(FRa)):
            if(FRa[j]<0):
                FRa[j]=0        
        if i==0:
            FR=FRa;
        FR=FR+FRa;
    return FR

@jit()
def FRGauss(spiketimes,resolution,fs,caus):
    """
    Author: Sergio Parra Sánchez
    This function creates the estimate in time of a firing rate dependent of time by
    using a Gaussian function as a kernel. 
    The kernel can be causal or not, depending whether the parameter caus is 1 or 0 respectively.
    
    For further information see Peter & Dayan, Theoretical Neuroscience, chapter 1
    
    """

    #from math imp   
    denominator=1/(sqrt(2*pi)*resolution);
    resolution=resolution**2;
    numel=ceil(spiketimes[-1]*fs);
    t=arange(0,numel/fs,1/fs);
    for i in range(0,size(spiketimes)):
        FRa=denominator*exp((-1*(t-spiketimes[i])**2)/(2*resolution));
        if(caus==1):
          f=(t>spiketimes[i]);
          FRa[f]=0;        
        if i==0:
            FR=FRa;
        FR=FR+FRa;
    return FR


@jit()
def FRcount(binary, delta, step, ind):
    """
     Author: Sergio Parra Sánchez
    This function creates the estimate in time of a firing rate dependent of time by
    the use of only event counts. This time window is causal.
    The parameters are:
        * 1 binary; a vector of ones and zeros to indicate the ocurrence of an event
        * 2 delta; the time length resolution, this must be in ms.  
        * 3 step; the windows spacing, is in ms.
        * 4 ind; this is a flag to indicate if input vector is any event vector or row vector.
    """
    r=size(binary);
    elements=0;   
    step=step*2;
    FR=zeros(r-delta);
    if(ind==0):
        delta=delta*2;
        begin=0;
        while(1):
            segment=binary[begin:begin+delta];
            FR[elements]=sum(segment);
            begin=begin+step;
            elements=elements+1;
            if begin>=r-delta:
                break;
    else:
        begin=binary[0];        
        while(1):
           count=(binary>=begin)*(binary<(begin+step))
           FR[elements]=sum(count)
           elements+=1
           begin+=step                     
    return FR[0:elements];

@jit
def sync_file(data, epoch, nrow, period):
    """
    Dada una lista de arreglos como se suelen cargar los archivos de espigas y el inicio y
    final de un intervalo, esta función devuelve el número de cuentas que están en ese intervalo para
    cada elemento de la lista.
    """
#    from numpy import zeros
#    from Phd import vecstrength
    sync=zeros((nrow, 2), dtype=float)
    for length in range(nrow):
         bolvec=zeros((len(data[length])), dtype=bool)
         bolvec[1::]=(data[length][1::]>=epoch[length, 0])*(data[length][1::]<=epoch[length, 1])
         theta, bartheta, sync[length, 1], sync[length, 0]=vecstrength(array(data[length])[bolvec], period)
#         theta, bartheta, R, vs
    return sync
#---------------------- Estos segmentos probablemente vayan a la compilación
    
@jit
def firing_file_wind(Data, Tiempos, nrow, window, step, informationfile, file):
    """
     Dada una lista de arreglos como se suelen cargar los archivos de espigas, así como el inicio y final
     de un intervalo para cada ensayo, esta función calculará el firing rate instantáneo utilizando una ventana
     cuadrada causal. Esta función pondrá los resultados en un archivo con extensión .csv. Para ello el archivo
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
   
    ncol=size(informationfile, axis=1)
    for rows in range(nrow):
        window_b=Tiempos[rows, 0]
        window_e=window_b + window
        counter=ncol
        Result=zeros((1, 1200), dtype=float)
        for col in range(ncol):
            Result[0, col]=informationfile[rows, col]
        while(window_e<=Tiempos[rows, 1]):
             try:
                bolvec=(array(Data[rows])[1::]>=window_b)*(array(Data[rows])[1::]<window_e)
             except:
                print("Aqui está el error")
             Result[0, counter]=nsum(bolvec)
             window_b+=step
             window_e=window_b+window
             counter+=1
        savetxt(file, Result[0:counter], delimiter=',', newline='\n')
        

@jit
def sync_file_wind(Data, Tiempos, nrow, Period, window, step, informationfile, file1, file2):
    """
     Dada una lista de arreglos como se suelen cargar los archivos de espigas, así como el inicio y final
     de un intervalo para cada ensayo, esta función calculará el vector strength utilizando una ventana
     cuadrada causal. Esta función pondrá los resultados en un archivo con extensión .csv. Para ello el archivo
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
    VS=zeros((1, 1200), dtype=float)
    Ray=zeros((1, 1200), dtype=float)
    ncol=size(informationfile, axis=1)
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

#@jit
def CV2(spike_trains, wnd,  stp, t0, tend):
    """
    Función realizada por Sergio Parra Sánchez
    Coeficiente de variación local, esta función calcula el coeficiente CV2 de acuerdo al articulo de :
    Ponce-Alvarez et al , Comparison of local measures of spike time irregularity and relating variability to firing rate in motor cortical neurons, 
    J Comput Neurosci, 2010,  DOI 10.1007/s10827-009-0158-2
    La función requiere los siguientes argumentos de entrada:
    * spikr_trains:   Trenes de espigas, es un numpy obj o bien una lista donde cada elemento es un numpy array con los tiempos de espigas.
    * wnd:  Tamaño de la ventana para la estimación del coeficiente de variación
    * stp: Paso de la ventana para la estimación del coeficiente de variación
    * t0: Tiempo inicial para el análisis
    * tend: TIempo final para el análisis
    """
    r=len(spike_trains);    
    micv2=[];
    mitim=[];
    limit=int(np.ceil((tend-t0)/stp)-wnd/stp+1);
    #print(limit)
    NTRIAL=np.zeros(( r, limit));
    NCV2=np.zeros( (r, limit));
    for i in range (r):  #Crea el raster de la cualidad local.
        ISI=np.diff(spike_trains[i].flatten());    
        #print("ISi calculado");
        micv2.append([np.array( 2 *(np.abs(ISI[1::]-ISI[0:-1]))/(ISI[1::] +ISI[0:-1])  )]);    
        mitim.append([np.array(  spike_trains[i][2::]) ]);       #Restas 2, 1 por el número de ensayo y otro por diff.  
        #Ahora comienza el promedio por ensayos.
        #print("Aquí comienza el promedio por ensayos")
        ti=t0; 
        column=0
        while(True):
            #print(column, "\t", ti)
            segment=(mitim[i][0][:, 0]>=ti)*(mitim[i][0][:, 0]<(ti +wnd));      
            NTRIAL[i, column]=np.sum(segment);
            #print(len(segment));
            NCV2[i, column ]=np.sum(micv2[i][0][segment]);
            ti+=stp;
            column+=1;
            if (ti+wnd)>tend: 
                break
    return NTRIAL, NCV2    
    


       
#a=fir_Inf_pfile(monkey=32, serie=164, whset=["setA", "setB", "setD"], met=2, guardar=1, graficar=0, periodo=["iE3", "fE3", 0, 0])
#a=psicofisica_archivos('D:\Drive_sparra\RS\S1_limpieza2.xlsx', 'D:\BaseDatosKarlitosNatsushiRR032\Text', 32, "Psych_RR032_S1.csv")
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------ ----------------------------------   
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------         
#N=optimumC(33, "RR033077_001", 4, 1, return_histogram=1) 
 

#R, perm=responsivity('RR032164_002', 3, 1, 'D:\BaseDatosKarlitosNatsushiRR032\Text_s', 'iE3', 2 ,0.050, 0.010, 1000)
#    
#file='RR032164_008'        
#elec, unit= 5, 1
#align='S3'
#limits=[-1, 10]
#Monkey=32
#savepath= 'D:\BaseDatosKarlitosNatsushiRR032\Text_s\RR032164_008'
#raster=S_RASTER(file, elec, unit, align, limits, Monkey, savepath)
##raster.uncertainty_v2()
##raster.attention()
#raster.focalization()
#mod=MODULATION('RR032164_001', 5, 1, 'S3', ['S3', 'F3'], 1)
#A=mod.FRmodulation()        
    
#
#cfile='RR032164_001'
#monkey=32
#electrode=5
#unit=2

#from os import name
#from numpy import loadtxt
#from pyexcel_ods import get_data
##  Carga el path para los archivos.
#if name=='nt':
#    if monkey==32:
#        cpath='D:\BaseDatosKarlitosNatsushiRR032\Text_s'
#    else:
#        cpath='D:\Database_RR033\Text_s'   
#else:
#    if monkey==32:
#        cpath='\run\media\AENHA\BaseDatosKarlitosNatsushiRR032\Text_s'
#    else:
#        cpath='\run\media\AENHA\Database_RR033\Text_s'  
#cfilex='%s\%s\%s_e%d_u%d.csv'%(cpath, cfile, cfile, electrode, unit)
#A=loadtxt(cfilex, usecols=0, dtype=int, delimiter=',')
#A.astype(int)
#A-=1
#Data=get_data(cfilex)
#cname="%s_e%d_u%d.csv"%(cfile, electrode, unit)
#Data=Data[cname]
#modulation=MODULATION(cfile, electrode, unit, [ ] ,['S3', 'F3'], 1)
#firing=modulation.FRmodulation()
#PSTH("RR032164_001", 5, 2000, 50, 2, "Protocol", "C:\\Users\\sparra\\Desktop", 1, 0 )   



#
###      
#monkey, cfile, electrode, unit, Nu=33, "RR033088_001", 1, 1, 3

#cname= 'D:\Drive_sparra\RS2\Raster33\set_i\\%s_%d'%(cfile, electrode)
#cname= '/run/media/sparra/AENHA/Drive_sparra/RS2/Raster33/Set_o'

#elRename_trials(cfile, electrode, unit, Nu, trials, monkey, 1)
#Combine_files(cfile, electrode, unit1, unit2, Nu, monkey)  
### Creación de gráficas raster 
#raster=S_RASTER(cfile, electrode, unit, 'S3', [-1,9.5], monkey, cname)
#raster.attention()
#raster.focalization()
##
