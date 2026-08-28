import matplotlib.pyplot as plt
import numpy as np

def chartEigenSig(gf):
    '''
    '''
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
    pltS(ax[0], gf.S)
    plotUV(ax[1],ax[2],gf)
    plt.tight_layout()
    plt.show()

def pltS(ax, S):
    '''
    '''
    ax.plot(S, '-rd', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('# of EigenValue')
    ax.set_ylabel('EigenValue')
    ax.tick_params(axis='y', labelrotation=90)    
    
def plotUV(ax3,ax4,gf):
   '''
    ax4 is V
    ax3 is U
   '''
   ax4.plot([x for x in range(gf.V.shape[1])], gf.V[0,:],  '-db', label='First EigenVector')
   ax4.set_xlabel('hour of the day')
   ax4.tick_params(axis='y', labelrotation=90)
   ax4.set_ylabel('nt')

   ax3.plot(gf.UU[:,0], '--dk', label='First Trace')
   ax3.plot(gf.dis, gf.U[:,0], 'or', label='First Trace')
   ax3.set_xlim([gf.dis[0], gf.dis[-1]])
   ax3.set_xlabel('# of Epoch')
   ax3.tick_params(axis='y', labelrotation=90)
   ax3.set_ylabel('load coef.')

def plotFre(lshe):
    '''
    '''
    lsheFreq = lshe.frequencies_
    coef     = lshe.weights_ 
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    P = np.zeros(len(lsheFreq), dtype=float) #,coef
    freq = np.array(lsheFreq)
    for i in range(P.shape[0]):
        P[i] = coef[2*(i+1)]**2+coef[2*(i+1)+1]**2
    idx = np.argsort(freq)
    F_sorted = freq[idx]
    P_sorted = P[idx]
    ax.set_title("detected frequencies")
    ax.plot(F_sorted,P_sorted, 'or', markersize=3 , label='Significant frequencies')
    ax.fill_between(F_sorted,0.0, P_sorted)
    ax.set_xlabel('frequency(1/hour)')
    ax.set_ylabel('Coef.')
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelrotation=90)
    ax.legend()

    plt.tight_layout()
    plt.show()  

def plotEigenSigFilled(df):
    '''
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(21, 5)) 
    ax[0].plot(df.index, df['SynthticGaps'].values,'-r', lw=1, label='Synthetic Gaps')
    ax[0].plot(df.index, df['EigenSigFilled'].values,'--k', lw= 0.75, label='Gap Filled')
    ax[1].plot(df.index, df['EigenSigResiduals'].values,'--k', lw= 0.75, label='Residuals')
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()