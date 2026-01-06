import numpy as np
import jax.numpy as jnp
from jax import vmap,grad,jit
from jax.tree_util import Partial
import optax

from .lddmm import LDDMMLoss
from .loss import VarifoldLoss
from .optimizer import BatchIteratedBarycenterOptimizer



####################################################################################################################################
####################################################################################################################################
### GENERAL ###
####################################################################################################################################
####################################################################################################################################

def _argmedian_length_from_mask(x):
    return np.argpartition(x, np.sum(x) // 2)[len(x) // 2]

def BarycenterLDDMMLoss(Kv,dataloss:callable,gamma_loss =0.,nt=10,deltat=1.0): 
    """ 
    Return the Barycenter LDDDMMLoss function related to the barycenter problem

    Parameters
    ----------
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        dataloss: (X,X_mask,Y,Y_mask)->float
            the attachment loss function, typically a varifold loss
        gamma: float 
            regularization constant related to the norm of the initial velocity
        nt: int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
            

    Returns
    -------
        LDDMM registration loss : (P,X,X_mask,Y,Y_mask)->float
         :math:`\\sum_i \\gamma |v_i|_{\\mathsf{V}}+\\mathcal{L}(\\phi^{v_i}\\cdot X,Y_i)` where X is an array of size (n_samples,d+1),
        and P are the momentums (n_samples,d+1) such that :math:`\\phi^{v_0}=\\exp_X(P)` is the exponential map which is computed by integrating the Hamiltonian system
        using the Ralston integrator on nt points. Also called shooting function.
    """
    unit_loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    f = vmap(unit_loss,(0,None,None,0,0),0)
    def bloss(p0,q0,q0_mask,q1,q1_mask): 
        return jnp.sum(f(p0,q0,q0_mask,q1,q1_mask))
    return bloss


# voir si on fait sauter time initializer
def batch_barycenter_registration(shoot,batched_q1,batched_q1_mask,Kv,dataloss:callable,init=None,init_mask=None,time_initializer=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,update_interval=100,verbose=True,stream_bool=False,stream_object=None):
    """ 
    Return the Barycenter LDDDMMLoss function related to the barycenter problem

    Parameters
    ----------
        shoot : (p0,q0,q0_mask)->(p1,q1)
            Shooting function, return the Hamiltonian dynamics at time 1 starting from (p0,q0)
        batched_q1 : array of shape (N_timeseries,batch_size,n_samples,d+1)
            target time series' graphs, in practice batch_size is set to 1
        batched_q1_mask : array of shape (N_timeseries,batch_size,n_samples,1)
            target time series' graphs mask
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        dataloss: (X,X_mask,Y,Y_mask)->float
            the attachment loss function, typically a varifold loss
        init : array of shape (N_timeseries,batch_size,n_samples,d+1)
            initial momentums for the optimizers, default None
        init_mask : array of shape (N_timeseries,batch_size,n_samples,1)
            initial momentums mask for the optimizers, default None
        niter : int
            number of iterations for the optimizer
        optimizer : optimize object from optax
            Default to optax.adabelief(learning_rate=0.1)
        gamma_loss: float 
            regularization constant related to the norm of the initial velocity
        nt: int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
        update_interval : int
            The barycenter is updated every update_interval steps by shooting the current barycental using the average of the estimated momentums,default 100
            

    Returns
    -------
        batched_p : array of shape (n_samples,d+1)
            momentums minimizing the LDDMM loss related to the multiple shooting problems
        q0 : array of shape (n_samples,d+1)
            learned barycenter = time series' graph of referece
        q0_mask : array of shape (n_samples,1)
            learned barycenter time series' graph mask
    """
    if init is None: 
        sigs_size = np.sum(batched_q1_mask,axis=2)
        median_length_idx = np.unravel_index(np.argsort(sigs_size.flatten(),kind='stable')[sigs_size.shape[0]//2],sigs_size.shape)[:-1]
        q0 = batched_q1[median_length_idx]
        q0_mask = batched_q1_mask[median_length_idx]
    else: 
        q0 = init
        q0_mask = init_mask
    if time_initializer is None:
        batched_p0 = jnp.zeros_like(batched_q1,dtype = jnp.float32)
    if callable(time_initializer): 
        if verbose: 
            print("Time initialization")
        batched_p0,q0,q0_mask = time_initializer(batched_q1,batched_q1_mask,q0,q0_mask)
    bloss = BarycenterLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchIteratedBarycenterOptimizer(shoot,bloss,niter,optimizer, update_interval =update_interval,verbose=verbose,stream_bool=stream_bool,stream_object=stream_object)
    return *opt(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask),q0_mask

def batch_varifold_barycenter_registration(batched_q1,batched_q1_mask,Kv,Kl,init=None,init_mask=None,time_initializer=None,niter=100, optimizer = optax.adam(learning_rate=0.1),gamma_loss =0.,nt=10,deltat=1.0,verbose=True):
    dataloss = VarifoldLoss(Kl)
    return batch_barycenter_registration(batched_q1,batched_q1_mask,Kv,dataloss,init,init_mask,time_initializer,niter,optimizer,gamma_loss,nt,deltat,verbose)
