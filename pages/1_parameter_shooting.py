import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import time


import sys
sys.path.insert(0,"../")

#import optax
#from optax.schedules import warmup_cosine_decay_schedule

from utils import Sampler,TASExperiment

from src.kernel import Velocity_TSLDDMM_Gaussian_Kernel
from src.lddmm import Shooting
from src.utils import time_shape_embedding
from src.loss import SumVarifoldLoss
import matplotlib.pyplot as plt

functional_signal = lambda x:  np.sin(2*np.pi*x)
#time series to time series'graph 
st.write(r"You can modify the parameter of TS-LDDMM kernel $K_G$ and the parameters of generation of $v_0$ to understand their effect on a geodesic shooting.")
q0 = time_shape_embedding(functional_signal(np.linspace(0,1,300)).reshape(-1,1),)
# représentation sous forme de graphe, maxlength pour générer un mask adaptaer
q0_mask = np.full_like(q0[:,:1],True).astype(bool) #mask is important when time series have different length
st.sidebar.write(r"You can modify the hyperparameters of the kernel $K_G$ defining the RKHS of the velocity field (eq 9).")
hyper_1=st.sidebar.slider(r"$c_{time}$", min_value =0.1, max_value =2.0, step=0.1,value=1.0)
hyper_2=st.sidebar.slider(r"$c_{space}$", min_value =0.05, max_value =1.0, step=0.05,value=0.1)
hyper_3=st.sidebar.slider(r"$\sigma_{time,0}$", min_value =60, max_value =140, step=1,value=60)
hyper_4=st.sidebar.slider(r"$\sigma_{time,1}$", min_value =0.5, max_value =5.0,step= 0.1,value=1.0)
hyper_5=st.sidebar.slider(r"$\sigma_{space,1}$",min_value = 0.5, max_value =5.0, step=0.1,value=1.0)
st.sidebar.write(r"(blue curve) You can choose the amplitude of the time coordinate ($t_a$)/space coordinate ($s_a$) of $v_0$ and how to regularize its coordinates ($m_s$) .")
t_a=st.sidebar.slider(r"$t_a$", min_value =0, max_value =50, step=1)
s_a=st.sidebar.slider(r"$s_a$", min_value =0, max_value =50, step=1)
m_a=st.sidebar.slider(r"$m_s$", min_value =10, max_value =100, step=10)

Kv = Velocity_TSLDDMM_Gaussian_Kernel(hyper_1,hyper_2,hyper_3,hyper_4,hyper_5) #return the kernel functions related to the prescribed hyper parameter
shoot = Shooting(Kv)# shooting functions taking p_0,q_0,q_0_mask

fig,ax = plt.subplots(1,1,figsize = (6,4))
ax.plot(*q0.T,color="tab:red", label=r"reference $s_0$")
t_lst = [
    [[t_a],[s_a],[m_a]],
    [[20],[20],[100]],
    [[5],[20],[10]]
]
plt.xlim(-350,350)
plt.ylim(-5,5)
for lst in t_lst:
    spl = Sampler(q0,q0_mask,shoot,*lst,0) # Specific to the project, to see the effect of the hyperparameter
    ps,qs,df= spl.rvs(1)
    ax.plot(*qs.T,label = r"$t_{a}$:" +f"{lst[0][0]}" r" - $s_{a}$:" + f"{lst[1][0]}" r" - $m_{s}$:" f"{lst[2][0]}")
ax.legend(fontsize=10)
ax.axhline(0,linestyle="--",color="black",zorder=0,linewidth=1)
fig.tight_layout()
st.pyplot(fig)



# spl = Sampler(q0,q0_mask,shoot,[10],[10],[50],0) #t_amp (norme pertubation temporel), s_amp (norme pertubation spatiale),smoothness, random state
# ps,qs,df= spl.rvs(10) #sampling de perturvation


# qs_mask = np.full_like(qs[:,:,:1],True).astype(bool)
# Kls = [TSGaussGaussKernel(2,1,2,0.6)] #kernel varifold
# dataloss = SumVarifoldLoss(Kls)
