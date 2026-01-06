
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import time
from optax.schedules import warmup_cosine_decay_schedule
import optax
import sys
sys.path.insert(0,"../")

#import optax
#from optax.schedules import warmup_cosine_decay_schedule

from utils import Sampler,TASExperiment

from src.kernel import Varifold_TSLDDMM_Gaussian_Kernel,Velocity_TSLDDMM_Gaussian_Kernel
from src.lddmm import Shooting, varifold_registration, Flowing
from src.utils import time_shape_embedding
from src.loss import VarifoldLoss,SumVarifoldLoss
# it can be interesting to use a sum of varifold and not only one varifold to tackle tail of distribution
from src.barycenter import batch_barycenter_registration
from src.utils import batch_dataset
from src.plotting import plot2Dfigure
import matplotlib.pyplot as plt

# Pour la prochaine fois, TODO

#Faire un dataset au bon format
#Essayer de faire tourner barycenter batch
# Demander à thibaut pourquoi batchsize =1 
# Afficher le barycentre et essayer de faire un shooting
# Afficher les éléments du dataset

# Episode 2, Faire une PCA en reprenant le notebook de Thibaut
# Episode 3, Montrer comment faire une régression logistique, voir avec Thibaut on intègre quoi?



np.random.seed(0)
st.write(r"You can modify the parameter of TS-LDDMM kernel $K_G$ and the varifold kernel $k_{pos}k_{dir}$ in order to estimate the barycenter. ")
# Création d'un dataset
N=7
t_sig = np.sin(np.linspace(0,1,200)*2*np.pi)
t_sig_=time_shape_embedding(t_sig.reshape(-1,1))

# We will have two types of variations, ratio frequency , and addition of some sinusoid
Dataset=np.zeros(((N-1)**2,200,2))
Dataset_mask=np.zeros(((N-1)**2,200,1))
for i in range(1,N):
    for j in range(1,N):
        t_sig_new=np.hstack((np.sin(np.linspace(0,0.5,j*10-1)*2*np.pi),np.sin(np.linspace(0.5,1,200-j*10+1)*2*np.pi)))
        t_sig_new[150:170] += i*2*np.sin(np.linspace(0,1,20)*2*np.pi)/20
        t_sig_new = t_sig_new.reshape(-1,1)
        t_sig_new=time_shape_embedding(t_sig_new)
        Dataset[(i-1)*(N-1)+j-1,:,:] = t_sig_new.copy()
        Dataset_mask[(i-1)*(N-1)+j-1,:,:] = np.full_like(t_sig_new[:,:1],True).astype(bool)
#n_ts,n_s,n_d = dataset.shape
sample_moyen=Dataset[(N//2-1)*(N-1)+N//2-1,:,:]
sample_moyen_mask=Dataset_mask[(N//2-1)*(N-1)+N//2-1,:,:]
bDataset,bDataset_mask = batch_dataset(Dataset,1,Dataset_mask)

st.sidebar.write(r"You can modify the hyperparameters of the kernel $K_G$ defining the RKHS of the velocity field (eq 9).")
hyper_1=st.sidebar.slider(r"$c_{time}$", min_value =0.1, max_value =2.0, step=0.1,value=1.0)
hyper_2=st.sidebar.slider(r"$c_{space}$", min_value =0.05, max_value =1.0, step=0.05,value=0.1)
hyper_3=st.sidebar.slider(r"$\sigma_{time,0}$", min_value =60, max_value =140, step=1,value=60)
hyper_4=st.sidebar.slider(r"$\sigma_{time,1}$", min_value =0.5, max_value =5.0,step= 0.1,value=1.0)
hyper_5=st.sidebar.slider(r"$\sigma_{space,1}$",min_value = 0.5, max_value =5.0, step=0.1,value=1.0)

Kv = Velocity_TSLDDMM_Gaussian_Kernel(hyper_1,hyper_2,hyper_3,hyper_4,hyper_5) #return the kernel functions related to the prescribed hyper parameter
shoot = Shooting(Kv)# shooting functions taking p_0,q_0,q_0_mask


st.sidebar.write(r"You can modify the hyperparameters of the Varifold kernel $K_{var}=k_{pos} k_{dir}$ (Appendix C.1).")
vari_hyper_1=st.sidebar.slider(r"$\sigma_{pos,t}$", min_value =0.1, max_value =4.0, step=0.1,value=2.0)
vari_hyper_2=st.sidebar.slider(r"$\sigma_{pos,x}$", min_value =0.1, max_value =4.0, step=0.1,value=1.0)
vari_hyper_3=st.sidebar.slider(r"$\sigma_{dir,t}$", min_value =0.1, max_value =4.0, step=0.1,value=2.0)
vari_hyper_4=st.sidebar.slider(r"$\sigma_{dir,x}$", min_value =0.1, max_value =4.0,step= 0.1,value=0.6)

Kl = Varifold_TSLDDMM_Gaussian_Kernel(vari_hyper_1,vari_hyper_2,vari_hyper_3,vari_hyper_4) #kernel varifold

#schedule =warmup_cosine_decay_schedule(0,0.1,10,100,0)
schedule =warmup_cosine_decay_schedule(0,0.3,80,800,0)
optimize=optax.adabelief(schedule)
dataloss=SumVarifoldLoss([Kl])# On peut rajouter d'autres noyaux si besoin

##registration ou varifold registration, barycentre iterated barycenter
latest_iteration = st.empty()
bar = st.progress(0)

import os
if not(os.path.exists("results/barycenter_p0s_N{:.0f}.npy".format(N))):
    p0s,q0,q0_mask = batch_barycenter_registration(shoot,bDataset,bDataset_mask,Kv,dataloss,niter=500,optimizer=optimize,gamma_loss=1e-3,stream_bool=True,stream_object=(bar,latest_iteration))

    np.save("results/barycenter_p0s.npy",p0s)
    np.save("results/barycenter_q0.npy",q0)
    np.save("results/barycenter_q0_mask.npy",q0_mask)
else: 
    p0s=np.load("results/barycenter_p0s.npy")
    q0=np.load("results/barycenter_q0.npy")
    q0_mask=np.load("results/barycenter_q0_mask.npy")

p0s = p0s.reshape(-1,p0s.shape[2],p0s.shape[3])
fig,ax = plt.subplots(1,1,figsize = (6,4))
print(q0.shape)
ax.plot(q0[:,0],q0[:,1],label="barycenter")
print(bDataset.shape)
for i in range(len(bDataset)):
    ax.plot(bDataset[i,0,:,0],bDataset[i,0,:,1],color="green",alpha=0.2)
ax.plot(sample_moyen[:,0],sample_moyen[:,1],label="sample moyen")
plt.legend()
fig.tight_layout()
st.pyplot(fig)
st.write(r"Barycenter learned on a synhtetic dataset in green")


