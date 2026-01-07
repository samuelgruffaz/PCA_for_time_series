import numpy as np
import jax.numpy as jnp
from jax import vmap 


def time_shape_embedding(x:np.ndarray,sampfreq=1,max_length=None,dtype = jnp.float32): 
    """time series to time series'graph 

    Parameters
    ----------
        x : (array) 
            signal (T,d) or ()
        sampfreq : (int, optional)
             sampling frequency. Defaults to 1. Important to set when it's not 1 for hyperparameter tuning
        max_length : (int, optional): 
            maximum lenght desired. Defaults to None.
        dtype : (_type_, optional):
             float32 or 64 depending on the memory. Defaults to jnp.float32.

    Returns
    -------
    pair
        The graph of the signal x : array of shape (T,d+1)
        Its mask  : array of shape (T,1)
    """
    n_s = x.shape[0]
    time = np.arange(n_s,dtype=float).reshape(-1,1)/sampfreq
    t_x = np.hstack((time,x),dtype=dtype)
    if max_length is None:
        return t_x
    else:
        t_x = np.pad(t_x,((0,max_length-n_s),(0,0))).astype(dtype)
        mask = np.full_like(t_x[:,:1],True,dtype=bool)
        mask[n_s:,:] = False
        return t_x, mask

def from_timeseries_to_dataset(X:list,sampfreq=1,dtype=jnp.float32): 
    lengths = np.array([x.shape[0] for x in X])
    max_length = np.max(lengths)
    mask_lst = []
    ts_lst = []
    for ts in X: 
        t_ts, t_mask = time_shape_embedding(ts,sampfreq,max_length,dtype=dtype)
        ts_lst.append(t_ts)
        mask_lst.append(t_mask)
    return jnp.array(ts_lst,dtype=dtype), jnp.array(mask_lst,dtype=jnp.bool_)
    
def batch_dataset(dataset,batch_size,masks=None,dtype=jnp.float32): 
    """
    Change the shape to apply a batch function

    Parameters
    ----------
        dataset : array of shape (n_ts,n_s,n_d)
            dataset of time series graph
        batch_size : (int)
            we should have n_ts // batch_size=int
        masks : array (n_ts,n_s,1) 
            Mask related to the dataset
       
    Raises
    ------
        ValueError: dataset size is not a multiple of batch_size

    Returns
    -------
    pair
        reshaped dataset : array of shape (n_batches,batch_size,n_s,n_d)
         
        reshaped mask : array of shape (n_batches,batch_size,n_s,1) 
    """
    if dataset.shape[0]%batch_size!=0: 
        raise ValueError("dataset size is not a multiple of batch_size")
    n_ts,n_s,n_d = dataset.shape
    n_batches = n_ts//batch_size
    if masks is None: 
        return dataset.reshape(n_batches,batch_size,n_s,n_d).astype(dtype)
    else: 
        return dataset.reshape(n_batches,batch_size,n_s,n_d).astype(dtype), masks.reshape(n_batches,batch_size,n_s,1)
    
def unbatch_dataset(batched_dataset,batched_masks=None,dtype=jnp.float32): 
    """
    Change the shape to remove batch dim

    Parameters
    ----------
        batched_dataset : array of shape (n_batches,batch_size,n_s,n_d)
            batch dataset of time series graph
    
        batched_masks : array (n_batches,batch_size,n_s,1)  
            Mask related to the dataset
       

    Returns
    -------
    pair
        dataset : array of shape (n,n_s,n_d)
         
        mask : array of shape (n,n_s,1) 
    """
    _,_,n_s,n_d = batched_dataset.shape
    if batched_masks is None: 
        return batched_dataset.reshape(-1,n_s,n_d).astype(dtype)
    else: 
        return batched_dataset.reshape(-1,n_s,n_d).astype(dtype), batched_masks.reshape(-1,n_s,1)
    
