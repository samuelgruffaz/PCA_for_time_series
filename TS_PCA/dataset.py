
import numpy as np
from .utils import time_shape_embedding




def generate_easy_dataset(N=8):
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

    graph_ref=Dataset[(N//2-1)*(N-1)+N//2-1,:,:]
    graph_ref_mask=Dataset_mask[(N//2-1)*(N-1)+N//2-1,:,:]
    return Dataset,Dataset_mask,graph_ref,graph_ref_mask

