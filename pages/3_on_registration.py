
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
from src.loss import SumVarifoldLoss
from src.plotting import plot2Dfigure
import matplotlib.pyplot as plt


st.write(r"You can modify the parameter of TS-LDDMM kernel $K_G$ and the varifold kernel $k_{pos}k_{dir}$ on the registration. The blue line is transformed in the red line to match the green line. The geodesic shooting problem is performed by minimizing a varifold loss using adabelief on 80 iterations with a cosinedecay.")
# Création du signal
t_sig = np.sin(np.linspace(0,1,200)*2*np.pi)
t_sig[150:170] += 1*np.sin(np.linspace(0,1,20)*2*np.pi)
t_sig = t_sig.reshape(-1,1)
t_sig = time_shape_embedding(t_sig)
t_sig_mask = np.full_like(t_sig[:,:1],True).astype(bool)





#t_sig et s_sig ont meme taille

s_sig = np.vstack((np.sin(np.linspace(0,0.5,59)*2*np.pi).reshape(-1,1),np.sin(np.linspace(0.5,1,141)*2*np.pi).reshape(-1,1)))
s_sig = time_shape_embedding(s_sig)
s_sig_mask = np.full_like(t_sig[:,:1],True).astype(bool)
print(len(t_sig),len(s_sig))

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



##registration ou varifold registration, barycentre iterated barycenter
latest_iteration = st.empty()
bar = st.progress(0)
p,q0,q0_mask=varifold_registration(s_sig,s_sig_mask,t_sig,t_sig_mask,Kv,Kl,niter=100,optimizer = optax.adabelief(warmup_cosine_decay_schedule(0,0.1,10,100,0)),stream_bool=True,stream_object=(bar,latest_iteration))
#p,q = varifold_registration(s_sig,t_sig,Kv,Kl)
#plot2Dfigure(s_sig,t_sig,p,Kv)
flow= Flowing(Kv,nt=10,deltat=1.) # TO UNDERSTAND, ASK THIBAUT
fig,axs=plot2Dfigure(s_sig,t_sig,p,shoot,flow,mask_s_sig=s_sig_mask,mask_t_sig=t_sig_mask )
# Si l'utilisateur peut mettre ses input il faudrait travailler avec un autre serveur et une autre app
st.pyplot(fig)

st.write(r"Looking at the grid before and after transformation you can observe whether the displacement is done along the time or the space axis.")