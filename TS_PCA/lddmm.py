import numpy as np 
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, grad, vmap, jacfwd
import optax
from .loss import VarifoldLoss
from .optimizer import RegistrationOptimizer, BatchOneToManyRegistrationOptimizer

# Comprendre le code, qu'est ce qui est TS-LDDMM est ce nécessaire d'être plus modulaire ?
# Notamment si l'on ne veut faire qu'une déformation temporel ?

# TO have only a deformation on time, we can consider a Kernel that (X,mask_X,Y,mask_Y,b)-> array of shape (n_samples,1) (1 to consider only time coordinate)
# The shape of momentum should be adapted 
# no problem as long as there is not matrix inversion


####################################################################################################################################
####################################################################################################################################
### GENERAL ###
####################################################################################################################################
####################################################################################################################################

def Hamiltonian(K:callable): 
    """ 
    Return the Hamiltonian function related to the kernel K

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
            

    Returns
    -------

        Hamiltonian function : (P,X,mask_X)-> float
        :math:`P^\\top K(X,X)P/2` where X is an array of size (n_samples,d+1), and :math:`K(X,X)` is the kernel matrix :math:`(k(x_i,x_j))`
        and P are the momentums, it's an array of size (n_samples,d+1)
    """
    def H(p,q,q_mask): 
        # p^\top K_q,q @p/2
       return 0.5*jnp.sum(p*K(q,q_mask,q,q_mask,p))
    return H 

def HamiltonianSystem(K:callable):
    """ 
    Return the Hamiltonian system function related to the kernel K

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
            

    Returns
    -------

        Hamiltonian system function : (P,X,mask_X)-> (-grad_H_pos,grad_H_momen)
        :math:`(-\\nabla_X H(X,P),\\nabla_P H(X,P))` where X is an array of size (n_samples,d+1), and :math:`K(X,X)` is the kernel matrix :math:`(k(x_i,x_j))`
        and P are the momentums, it's an array of size (n_samples,d+1)
    """
    H = Hamiltonian(K)
    def HS(p,q,q_mask):
        Gp,Gq = grad(H,(0,1))(p,q,q_mask)
        return -Gq,Gp
    return HS

def Shooting(K:callable,nt=10,deltat=1.):
    """ 
    Return the Hamiltonian system function related to the kernel K

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        nt: int
            number of integration time points
        deltat: float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt
            

    Returns
    -------

        Exponential flow function : (P,X,mask_X)-> (Exp_X(deltat P),Exp_X'(deltat P))
        :math:`(\\exp_X(\\delta P),\\exp_X'(\\delta P))` where X is an array of size (n_samples,d+1),
        and P are the momentums, it's an array of size (n_samples,d+1). The exponential map is computed by integrating the Hamiltonian system
        using the Ralston integrator. Also called shooting function.
    """
    
    HSystem = HamiltonianSystem(K)
    dt = deltat/nt
    def body_function(i,arg): 
        p,q,q_mask = arg
        dp,dq = HSystem(p,q,q_mask)
        pi,qi = p+(2*dt/3)*dp,q+(2*dt/3)*dq
        dpi,dqi = HSystem(pi,qi,q_mask)
        p,q = p+0.25*dt*(dp+3*dpi),q+0.25*dt*(dq+3*dqi)
        return p,q,q_mask
    def f(p,q0,q0_mask):
        p,q0,q0_mask = lax.fori_loop(0,nt,body_function,(p,q0,q0_mask))
        return p,q0
    return f



def LDDMMLoss(K:callable,dataloss:callable,gamma =0.001,nt=10,deltat=1.0): 
    """ 
    Return the Loss function related to a registration problem

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
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
        :math:`\\gamma |v_0|_{\\mathsf{V}}+\\mathcal{L}(\\phi^{v_0}\\cdot X,Y)` where X is an array of size (n_samples,d+1),
        and P are the momentums, it's an array of size (n_samples,d+1). :math:`\\phi^{v_0}=\\exp_X(P)` is the exponential map which is computed by integrating the Hamiltonian system
        using the Ralston integrator on nt points. Also called shooting function.
    """
    Hm = Hamiltonian(K)
    shoot =Shooting(K,nt,deltat)
    def loss(p0,q0,q0_mask,q1,q1_mask): 
        p,q = shoot(p0,q0,q0_mask)
        return gamma * Hm(p0,q0,q0_mask) + dataloss(q,q0_mask,q1,q1_mask)
    return loss

def Flowing(K:callable,nt=10,deltat=1.): 
    """ 
    Return the Exponential map function related to the kernel K that map as well a grid

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        nt: int
            number of integration time points
        deltat: float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt
            

    Returns
    -------

        Exponential flow function, taking a grid as well in input : (Grid,P,X,mask_X)-> (Grid_shooted,\Exp_X(deltat P),\Exp_X'(deltat P))
        :math:`(Grid_shooted,\\exp_X(\\delta P),\\exp_X'(\\delta P))` where X is an array of size (n_samples,d+1),
        and P are the momentums, it's an array of size (n_samples,d+1),
        Grid is another array that may be different than X, Grid_shooted is its final position after flowing by using the integrated velocity. The exponential map is computed by integrating the Hamiltonian system
        using the Ralston integrator. Also called shooting function.
    """
    
    HSystem = HamiltonianSystem(K)
    def FSystem(x,p,q,q_mask): 
        x_mask = jnp.full_like(x[:,:1],True,dtype=np.bool_)
        return (K(x,x_mask,q,q_mask,p),) + HSystem(p,q,q_mask) # On concatene les tuples pour rajouter un moment adapté au x, c'est pour la grille
    dt = deltat/nt
    def body_function(i,arg): 
        x,p,q,q_mask = arg
        dx,dp,dq = FSystem(x,p,q,q_mask)
        xi,pi,qi = x+(2*dt/3)*dx,p+(2*dt/3)*dp,q+(2*dt/3)*dq
        dxi,dpi,dqi = FSystem(xi,pi,qi,q_mask)
        x,p,q = x+0.25*dt*(dx+3*dxi),p+0.25*dt*(dp+3*dpi),q+0.25*dt*(dq+3*dqi)
        return x,p,q,q_mask
    def f(x0,p0,q0,q0_mask):
        x,p,q,q_mask = lax.fori_loop(0,nt,body_function,(x0,p0,q0,q0_mask))
        return x,p,q 
    return f



def DeformationGradient(K:callable,nt=10,deltat=1.0): 
    """ 
    Return the gradient of the exponential map according to the starting points

    Parameters
    ----------

        Kernel function : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        nt: int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
            

    Returns
    -------

        Gradient : (P,X,X_mask)->Jacobian_position(Shooting(K,nt,deltat))
        :math:`\\nabla_X \\exp_X(P)` where X is an array of size (n_samples,d+1),
        and P are the momentums, it's an array of size (n_samples,d+1). :math:`\\exp_X` is the exponential map which is computed by integrating the Hamiltonian system
        using the Ralston integrator on nt points. Also called shooting function.

    """
    jacHri = jacfwd(Shooting(K,nt,deltat),1)
    def gradient(p0,q0,q0_mask): 
        gr = jacHri(p0,q0,q0_mask)[1]
        idx = np.arange(gr.shape[0])
        gr = gr[idx,:,idx,:]-1
        return gr
    return jit(gradient)

#stream_bool
#stream_object for streamlit
#optax is the best 


def registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,p0 = None,gamma_loss=0.0,niter=100,optimizer = optax.adabelief(learning_rate=0.1),nt=10,deltat=1.0,verbose=True,stream_bool=False,stream_object=None):
    """ 
    Return the momentum p such that the hamiltonian dynamics starting from (q_0,p) reach approximately (q_1,p_1) at time t=1, the shooting problem is performed by minimizing the LDDMM loss defined previously

    Parameters
    ----------

        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
        q1 : array of shape (n_samples,d+1)
            target time series' graph
        q1_mask : array of shape (n_samples,1)
            target time series' graph mask
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        dataloss: (X,X_mask,Y,Y_mask)->float
            the attachment loss function, typically a varifold loss
        p0 : array of shape (n_samples,d+1)
            initial time series' graph momentum (facultative)
        gamma_loss: float 
            regularization constant related to the norm of the initial velocity
        niter : int
            number of iterations for the optimizer
        optimizer : optimize object from optax
            Default to optax.adabelief(learning_rate=0.1)
        nt : int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
        verbose : Boolean
            Default to True, print loss every 10 iterations
        
            

    Returns 
    -------

        p : array of shape (n_samples,d+1)
             initial momentum minimizing the LDDMM loss related to the shooting problem
        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
    """
    loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = RegistrationOptimizer(loss,niter,optimizer,verbose=verbose,stream_bool=stream_bool,stream_object=stream_object)
    if p0 is None:
        p0 = jnp.zeros_like(q0)
    if callable(p0):
        if verbose:
            print("Time initialization")
        p0 = p0(q0,q0_mask,q1,q1_mask)
    p = opt(p0,q0,q0_mask,q1,q1_mask)
    return p,q0,q0_mask



#The following function should be used in a class in a second time

def varifold_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True,stream_bool=False,stream_object=None): 
    """
    Return the momentum p such that the hamiltonian dynamics starting from (q_0,p) reach approximately (q_1,p_1) at time t=1, the shooting problem is performed by minimizing the LDDMM loss defined previously

    Parameters
    ----------

        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
        q1 : array of shape (n_samples,d+1)
            target time series' graph
        q1_mask : array of shape (n_samples,1)
            target time series' graph mask
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        Kl : (X,mask_X,Y,mask_Y,b)-> array of the shape of b
            The Varifold kernel function. :math:`K(X,Y)b` where X and Y are array of size (n_samples,d+1), :math:`K(X,Y)` is the kernel matrix :math:`(k(x_i,y_j))`
            and b is an array of shape (n_samples,d) with d the dimension of the problem
        p0 : array of shape (n_samples,d+1)
            initial time series' graph momentum (facultative)
        gamma_loss: float 
            regularization constant related to the norm of the initial velocity
        niter : int
            number of iterations for the optimizer
        optimizer : optimize object from optax
            Default to optax.adabelief(learning_rate=0.1)
        nt : int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
        verbose : Boolean
            Default to True, print loss every 10 iterations
        
    Returns 
    -------

        p : array of shape (n_samples,d+1)
             initial momentum minimizing the LDDMM loss related to the shooting problem using the varifold loss as data attachment term
        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
    """
    dataloss = VarifoldLoss(Kl)
    return registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,p0,gamma_loss,niter,optimizer,nt,deltat,verbose=verbose,stream_bool=stream_bool,stream_object=stream_object)


def batch_one_to_many_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True,stream_bool=True,stream_object=None):
    """
    Return the momentums (p^i) such that the hamiltonian dynamics starting from any (q_0,p_i) reach approximately (q_1,p^i_1) at time t=1, the shooting problem is performed by minimizing the LDDMM loss defined previously

    Parameters
    ----------

        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
        batched_q1 : array of shape (N_timeseries,batch_size,n_samples,d+1)
            target time series' graphs, in practice batch_size is set to 1
        batched_q1_mask : array of shape (N_timeseries,batch_size,n_samples,1)
            target time series' graphs mask
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        dataloss: (X,X_mask,Y,Y_mask)->float
            the attachment loss function, typically a varifold loss
        p0 : array of shape (n_samples,d+1)
            initial time series' graph momentum (facultative)
        gamma_loss: float 
            regularization constant related to the norm of the initial velocity
        niter : int
            number of iterations for the optimizer
        optimizer : optimize object from optax
            Default to optax.adabelief(learning_rate=0.1)
        nt : int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
        verbose : Boolean
            Default to True, print loss every 10 iterations
        
    Returns 
    -------

        batched_p : array of shape (n_samples,d+1)
            momentums minimizing the LDDMM loss related to the multiple shooting problems
        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
    """
    loss = LDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
    opt = BatchOneToManyRegistrationOptimizer(loss,niter,optimizer,verbose=verbose,stream_bool=stream_bool,stream_object=stream_object)
    if batched_p0 is None:
        batched_p0 = jnp.zeros_like(batched_q1)
    if callable(batched_p0):
        batched_p0 = batched_p0(q0,q0_mask,batched_q1,batched_q1_mask)
    batched_p = opt(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask)
    return batched_p,q0,q0_mask



# same should be used within a class
def batch_one_to_many_varifold_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True,stream_bool=True,stream_object=None):
    """
    Return the momentums (p^i) such that the hamiltonian dynamics starting from any (q_0,p_i) reach approximately (q_1,p^i_1) at time t=1, the shooting problem is performed by minimizing the LDDMM loss defined previously

    Parameters
    ----------

        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
        batched_q1 : array of shape (N_timeseries,batch_size,n_samples,d+1)
            target time series' graphs, in practice batch_size is set to 1
        batched_q1_mask : array of shape (N_timeseries,batch_size,n_samples,1)
            target time series' graphs mask
        Kv : (X,mask_X,Y,mask_Y,b)-> shape of b
            The Velocity kernel function (ideally the one given by TS-LDDMM)
        Kl : (X,mask_X,Y,mask_Y,b)-> array of the shape of b
            The Varifold kernel function. :math:`K(X,Y)b` where X and Y are array of size (n_samples,d+1), :math:`K(X,Y)` is the kernel matrix :math:`(k(x_i,y_j))`
            and b is an array of shape (n_samples,d) with d the dimension of the problem
        p0 : array of shape (n_samples,d+1)
            initial time series' graph momentum (facultative)
        gamma_loss: float 
            regularization constant related to the norm of the initial velocity
        niter : int
            number of iterations for the optimizer
        optimizer : optimize object from optax
            Default to optax.adabelief(learning_rate=0.1)
        nt : int
            number of integration time points, default=10
        delta : float 
            total time of integration :math:`\\delta`, stepsize=deltat/nt, default to 1
        verbose : Boolean
            Default to True, print loss every 10 iterations
        
    Returns 
    -------

        batched_p : array of shape (n_samples,d+1)
             initial momentums minimizing the LDDMM loss related to the multiple shooting problems
        q0 : array of shape (n_samples,d+1)
            initial time series' graph
        q0_mask : array of shape (n_samples,1)
            initial time series' graph mask
    """
    dataloss = VarifoldLoss(Kl)
    return batch_one_to_many_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0,gamma_loss,niter,optimizer,nt,deltat,verbose,stream_bool=stream_bool,stream_object=stream_object)
# def Flowing(K:callable,nt=10,deltat=1.): 
#     Fri = FlowRalstonIntegrator(K,nt,deltat)
#     def flow(x0,p0,q0,q0_mask): 
#         return Fri(x0,p0,q0,q0_mask)
#     return flow
#def Shooting(K:callable,nt=10,deltat=1.0): 
#     """
#     - K : kernel function (timeseries graph,mask)^2x(moment,mask)->(velocity,mask)
#     - nt: number of time points 
#     - delta: total time of integration, stepsize= delta/nt

#     Return a function that tale into account initial position and momentum for Exp(1) shooting
#     """ 
#     Hri = HamiltonianRalstonIntegrator(K,nt,deltat)
#     def shoot(p0,q0,q0_mask): 
#         return Hri(p0,q0,q0_mask)
#     return shoot

####################################################################################################################################
####################################################################################################################################
### TIME ###
####################################################################################################################################
####################################################################################################################################

# def TimeLDDMMLoss(K:callable,dataloss:callable,gamma =0.001,nt=10,deltat=1.0): 
#     Hm = Hamiltonian(K)
#     shoot =Shooting(K,nt,deltat)
#     def loss(p0,q0,q0_mask,q1,q1_mask): 
#         t_q0, s_q0 = q0[:,:1],q0[:,1:]
#         p,t_q = shoot(p0,t_q0,q0_mask)
#         return gamma * Hm(p0,t_q0,q0_mask) + dataloss(jnp.hstack((t_q,s_q0)),q0_mask,q1,q1_mask)
#     return loss


# def TimeShooting(K:callable,nt=10,deltat=1.0): 
#     Hri = HamiltonianRalstonIntegrator(K,nt,deltat)
#     def shoot(p0,q0,q0_mask): 
#         t_q0,s_q0 = q0[:,:1],q0[:,1:]
#         p,t_q =Hri(p0,t_q0,q0_mask)
#         q = jnp.hstack((t_q,s_q0))
#         return p,q
#     return shoot

# def TimeFlowing(K:callable,nt=10,deltat=1.): 
#     Fri = FlowRalstonIntegrator(K,nt,deltat)
#     def flow(x0,p0,q0,q0_mask): 
#         t_x0,s_x0 = x0[:,:1],x0[:,1:]
#         t_q0,s_q0 = q0[:,:1],q0[:,1:]
#         t_x,p,t_q = Fri(t_x0,p0,t_q0,q0_mask)
#         x = jnp.hstack((t_x,s_x0))
#         q = jnp.hstack((t_q,s_q0))
#         return x,p,q
#     return flow

# def TimeDeformationGradient(K:callable,nt=10,deltat=1.0): 
#     jacHri = jacfwd(HamiltonianRalstonIntegrator(K,nt,deltat),1)
#     def gradient(p0,q0,q0_mask): 
#         t_q0,s_q0 = q0[:,:1],q0[:,1:]
#         gr =jacHri(p0,t_q0,q0_mask)[1]
#         return np.diag(gr[:,0,:,0]).reshape(-1,1)-1
#     return gradient

# def time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
#     loss = TimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
#     opt = RegistrationOptimizer(loss,niter,optimizer,verbose)
#     p0 = jnp.zeros_like(q0[:,:1])
#     p = opt(p0,q0,q0_mask,q1,q1_mask)
#     return p,q0,q0_mask

# def varifold_time_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,gamma_loss=0.001,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
#     dataloss = VarifoldLoss(Kl)
#     return time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,gamma_loss,niter,optimizer,nt,deltat,verbose)

# def batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
#     loss = TimeLDDMMLoss(Kv,dataloss,gamma_loss,nt,deltat)
#     opt = BatchOneToManyRegistrationOptimizer(loss,niter,optimizer,verbose=verbose)
#     if batched_p0 is None:
#         batched_p0 = jnp.zeros_like(batched_q1[:,:,:,:1])
#     batched_p = opt(batched_p0,q0,q0_mask,batched_q1,batched_q1_mask)
#     return batched_p,q0,q0_mask

# def batch_one_to_many_varifold_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,batched_p0=None,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=True):
#     dataloss = VarifoldLoss(Kl)
#     return batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,batched_p0,gamma_loss,niter,optimizer,nt,deltat,verbose)

# def time_initializer(Kv,dataloss,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
#     def initializer(q0,q0_mask,q1,q1_mask):
#         p,q,qm = time_registration(q0,q0_mask,q1,q1_mask,Kv,dataloss,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
#         return jnp.pad(p,((0,0),(0,q.shape[1]-1)))
#     return initializer

# def varifold_time_initializer(Kv,Kl,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
#     def initializer(q0,q0_mask,q1,q1_mask):
#         p,q,qm = varifold_time_registration(q0,q0_mask,q1,q1_mask,Kv,Kl,gamma_loss,niter,optimizer,nt,deltat,verbose)
#         return jnp.pad(p,((0,0),(0,q.shape[1]-1)))
#     return initializer

# def batch_time_initializer(Kv,dataloss,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
#     def initializer(q0,q0_mask,batched_q1,batched_q1_mask):
#         bp,bq,bqm = batch_one_to_many_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,dataloss,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
#         return jnp.pad(bp,((0,0),(0,0),(0,0),(0,bq.shape[1]-1)))
#     return initializer

# def batch_varifold_time_initializer(Kv,Kl,gamma_loss=0.0,niter=100,optimizer = optax.adam(learning_rate=0.1),nt=10,deltat=1.0,verbose=False):
#     def initializer(q0,q0_mask,batched_q1,batched_q1_mask):
#         bp,bq,bqm = batch_one_to_many_varifold_time_registration(q0,q0_mask,batched_q1,batched_q1_mask,Kv,Kl,None,gamma_loss,niter,optimizer,nt,deltat,verbose)
#         return jnp.pad(bp,((0,0),(0,0),(0,0),(0,bq.shape[1]-1)))
#     return initializer





