
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
from src.lddmm import Shooting, varifold_registration, Flowing,batch_one_to_many_varifold_registration
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
N=8
t_sig = np.sin(np.linspace(0,1,200)*2*np.pi)


# We will have two types of variations, ratio frequency , and addition of some sinusoid
Dataset=np.zeros(((N-1)**2,200,2))
Dataset_mask=np.zeros(((N-1)**2,200,1))
Feature=np.zeros(((N-1)**2,2))
for i in range(1,N):
    for j in range(1,N):
        t_sig_new=np.hstack((np.sin(np.linspace(0,0.5,j*10-1)*2*np.pi),np.sin(np.linspace(0.5,1,200-j*10+1)*2*np.pi)))
        t_sig_new[150:170] += i*2*np.sin(np.linspace(0,1,20)*2*np.pi)/20
        t_sig_new = t_sig_new.reshape(-1,1)
        t_sig_new=time_shape_embedding(t_sig_new)
        Dataset[(i-1)*(N-1)+j-1,:,:] = t_sig_new.copy()
        Dataset_mask[(i-1)*(N-1)+j-1,:,:] = np.full_like(t_sig_new[:,:1],True).astype(bool)
        Feature[(i-1)*(N-1)+j-1]=np.array([i,j])
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

niter=100
schedule =warmup_cosine_decay_schedule(0,0.1,10,100,0)
#schedule =warmup_cosine_decay_schedule(0,0.3,80,800,0)
optimize=optax.adabelief(schedule)
dataloss=SumVarifoldLoss([Kl])# On peut rajouter d'autres noyaux si besoin

##registration ou varifold registration, barycentre iterated barycenter
latest_iteration = st.empty()
bar = st.progress(0)
import os
#p0s,q0,q0_mask = batch_barycenter_registration(bDataset,bDataset_mask,Kv,dataloss,niter=500,optimizer=optimize,gamma_loss=1e-3,stream_bool=True,stream_object=(bar,latest_iteration))
if not(os.path.exists("results/registration_p0s_N{:.0f}.npy".format(N))):
    p0s,q0,q0_mask= batch_one_to_many_varifold_registration(sample_moyen,sample_moyen_mask,bDataset,bDataset_mask,Kv,Kl,niter=niter,optimizer=optimize,gamma_loss=1e-3,stream_bool=True,stream_object=(bar,latest_iteration))

#st.pyplot(fig)
    np.save("results/registration_p0s_N{:.0f}.npy".format(N),p0s)
    np.save("results/registration_q0_N{:.0f}.npy".format(N),q0)
    np.save("results/registration_q0_mask_N{:.0f}.npy".format(N),q0_mask)
else: 
    p0s=np.load("results/registration_p0s_N{:.0f}.npy".format(N))
    q0=np.load("results/registration_q0_N{:.0f}.npy".format(N))
    q0_mask=np.load("results/registration_q0_mask_N{:.0f}.npy".format(N))
p0s = p0s.reshape(-1,p0s.shape[2],p0s.shape[3])
fig,ax = plt.subplots(1,1,figsize = (6,4))
ax.plot(q0[:,0],q0[:,1])

# Rajouter la PCA

idx = 2
p,q = shoot(p0s[idx],q0,q0_mask)


x = bDataset[idx,0] # 0 for un batch

qp = q #use mask if necessary
ax.plot(x[:,0],x[:,1],"--",linewidth=5,alpha=0.5,label="sample")
ax.plot(qp[:,0],qp[:,1],"-",alpha=1,label='after_shooting')
ax.legend(fontsize=10)

fig.tight_layout()
st.pyplot(fig)
st.write(r"Registration using the mean sample as barycenter")

print(q0.shape,q0_mask.shape) # (n_sample,d),(n_sample,1)


ft_size = 25
pft_size = 20
from src.statistic import MomentaPCA
n_comp,n_disp = 2,3
components = [f"PC{i+1}" for i in range(n_comp)]
mpca = MomentaPCA(n_comp,False,"cov")
mpca.fit(Kv,p0s,q0,q0_mask)

p0_bar = mpca.m_ps_ #mean p0
plot_mask = np.where(q0_mask==True)[0]
k = 1
shoot = Shooting(Kv)
p0_b = mpca.m_ps_
fig2,axs = plt.subplots(n_comp,n_disp,figsize = (n_disp*3,n_comp*3),sharex=True,sharey=True)
for pca_index in range(n_comp):
    sigma_pca = mpca.p_std_[pca_index]
    for j,alpha in enumerate(np.linspace(-2*sigma_pca , 2*sigma_pca, n_disp)):
        p0_mode = p0_b + alpha * mpca.p_pc_[pca_index]
        _,q = shoot(p0_mode,q0,q0_mask)
        plot_q = q[plot_mask].T
        axs[pca_index,j].plot(plot_q[0],plot_q[1],color="tab:red")
        #axs[pca_index,j].plot(plot_q[0],plot_q[2])
    axs[pca_index,0].set_ylabel(components[pca_index],fontsize=ft_size)



for ax in axs.flatten():
    ax.axhline(0,color="black",linewidth=1,linestyle="--",zorder=0)

axs[-1,1].set_xlabel("time (ms)",fontsize=pft_size)

axs[0,0].set_title(r"-2$\sigma_{PC}$",fontsize=ft_size)
axs[0,1].set_title(r"$\mathsf{G}_0$",fontsize=ft_size)
axs[0,2].set_title(r"+2$\sigma_{PC}$",fontsize=ft_size)
axs[0,2].legend(fontsize = pft_size,loc=1)
for ax in axs[-1,:]:
    ax.xaxis.set_tick_params(labelsize=pft_size)
for ax in axs[:,0]:
    ax.yaxis.set_tick_params(labelsize=pft_size)

#fig.suptitle("II) PC shootings",fontsize = ft_size)
fig2.suptitle(" ",fontsize = ft_size)

fig2.subplots_adjust(left=0.1,bottom=0.15,right=0.9,top=0.85,wspace=0.1,hspace=0.15)
fig2.tight_layout()
st.pyplot(fig2)
st.write(r"Représentation des composantes kernelPCA associé à la représentation TS-LDDMM")
Feature_centered=Feature-Feature.mean(axis=0)
coord=mpca.p_score_
fig3,ax = plt.subplots(1,1,figsize = (6,4))
ax.scatter(coord[:,0],coord[:,1],s=1+10*Feature[:,0], c=Feature[:,1]/len(Feature), cmap='viridis')#fontsize=Feature_centered[:,0]
# from matplotlib import cm
# viridis = cm.get_cmap('viridis')
# cbar = plt.colorbar(viridis, ax=ax)
#ax.plot(Feature_centered[:,0],Feature_centered[:,1],'+')
ax.set_xlabel('PC_0')
ax.set_ylabel('PC_1')


fig3.tight_layout()
st.pyplot(fig3)
st.write(r"Les tailles de points varient en fonction du ratio de taille entre première et seconde bosse.")
st.write(r"Les couleurs varient en fonction de la taille de la petite sinusoide à droite.")

st.write(r"**L'exemple suivant fait la même chose avec PCA directement sur les signaux**")
from sklearn.decomposition import PCA

pca = PCA(n_components=n_comp)

n_data,_,_=Dataset.shape
X_D=np.reshape(Dataset,(n_data,-1))
pca.fit(X_D)
print(pca.singular_values_)
print(X_D.shape)
Coord_pca=pca.transform(X_D)
print(Coord_pca)
fig4,ax = plt.subplots(1,1,figsize = (6,4))
ax.scatter(Coord_pca[:,0],Coord_pca[:,1],s=1+10*Feature[:,0], c=Feature[:,1]/len(Feature), cmap='viridis',alpha=0.5)#fontsize=Feature_centered[:,0]
# from matplotlib import cm
# viridis = cm.get_cmap('viridis')
# cbar = plt.colorbar(viridis, ax=ax)
#ax.plot(Feature_centered[:,0],Feature_centered[:,1],'+')
ax.set_xlabel('PC_0')
ax.set_ylabel('PC_1')
fig4.tight_layout()
st.pyplot(fig4)
st.write(r"Les coordonnés se superposent car les composantes apprennent difficilement les variations.")
q0_b=np.reshape(X_D.mean(axis=0),(-1,2))

fig5,axs = plt.subplots(n_comp,n_disp,figsize = (n_disp*3,n_comp*3),sharex=True,sharey=True)
for pca_index in range(n_comp):
    sigma_pca = pca.singular_values_[pca_index]
    for j,alpha in enumerate(np.linspace(-2*sigma_pca , 2*sigma_pca, n_disp)):
        q0_mode = q0_b + alpha * np.reshape(pca.components_[pca_index],(-1,2))
        
        axs[pca_index,j].plot(q0_mode[:,0],q0_mode[:,1],color="tab:red")
        #axs[pca_index,j].plot(plot_q[0],plot_q[2])
    axs[pca_index,0].set_ylabel(components[pca_index],fontsize=ft_size)



for ax in axs.flatten():
    ax.axhline(0,color="black",linewidth=1,linestyle="--",zorder=0)

axs[-1,1].set_xlabel("time (ms)",fontsize=pft_size)

axs[0,0].set_title(r"-2$\sigma_{PC}$",fontsize=ft_size)
axs[0,1].set_title(r"$\mathsf{G}_0$",fontsize=ft_size)
axs[0,2].set_title(r"+2$\sigma_{PC}$",fontsize=ft_size)
axs[0,2].legend(fontsize = pft_size,loc=1)
for ax in axs[-1,:]:
    ax.xaxis.set_tick_params(labelsize=pft_size)
for ax in axs[:,0]:
    ax.yaxis.set_tick_params(labelsize=pft_size)

#fig.suptitle("II) PC shootings",fontsize = ft_size)
fig5.suptitle(" ",fontsize = ft_size)

fig5.subplots_adjust(left=0.1,bottom=0.15,right=0.9,top=0.85,wspace=0.1,hspace=0.15)
fig5.tight_layout()
st.pyplot(fig5)



#batch_one_to_many_varifold_registration si on a deja le barycentre on peut utiliser cette fonction voir comment Thibaut l'utilisait


# np.random.seed(0)

# y =pd.read_csv("./dataset/y_600.csv",index_col=0)
# X = np.load("./dataset/X_600.npy")
# X_mask = np.load("./dataset/X_600_mask.npy")
# idxs = np.load("./results/ts_lddmm_exp_1/idxs.npy") # les indices qu'il faut selection

# X = X[:,::2,:] # prend un élément sur 2 pour la deuxième dimension
# X_mask = X_mask[:,::2,:]
# bX,bX_mask = batch_dataset(X[idxs],1,X_mask[idxs])
# y = y.iloc[idxs] # label sur les genotypes

# print(y.before.unique())
# print(y.genotype.unique())


# if __name__ == "__main__": 

#     Kv = VFTSGaussKernel(1,0.1,150,1,2)
#     Kl1 = TSGaussGaussKernel(5,2,5,1)
#     Kl2 = TSGaussGaussKernel(2,1,2,0.6)
#     Kl3 = TSGaussGaussKernel(1,0.6,1,0.6)
#     Kls=[Kl1,Kl2,Kl3]
#     dataloss = SumVarifoldLoss(Kls)
#     schedule = warmup_cosine_decay_schedule(0,0.3,80,800,0)
#     optimizer = optax.adabelief(schedule)



#     for i,filename in enumerate(y.filename.unique()): 
       
#         print(f"Mouse: {i+1}/{y.filename.unique().shape[0]} -- {filename}")
#         tidxs = y[y.filename == filename].index
#         bX,bX_mask = batch_dataset(X[tidxs],1,X_mask[tidxs]) #mouse per mouse, weird
#         p0s,q0,q0_mask = batch_barycenter_registration(bX,bX_mask,Kv,dataloss,niter=500,optimizer=optimizer,gamma_loss=1e-3)

#         np.save("./results/ts-lddmm_exp_1/"+f"{filename[:-4]}_p0s.npy",p0s)
#         np.save("./results/ts-lddmm_exp_1/"+f"{filename[:-4]}_q0.npy",q0)
#         np.save("./results/ts-lddmm_exp_1/"+f"{filename[:-4]}_q0_mask.npy",q0_mask)

#     print("Done")