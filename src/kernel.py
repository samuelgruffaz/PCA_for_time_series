import numpy as np
import jax.numpy as jnp

#making more modular using a triangular structure + a kernel ? Not necessary


def Varifold_TSLDDMM_Gaussian_Kernel(t_sigma_1:float,s_sigma_1:float,t_sigma_2:float,s_sigma_2:float,np_dtype = np.float32):
    """
    Generate TS-LDDMM  Varifold kernel function,  using the notations of the paper "Shape analysis for time series"
    :math:`k((x,v),(x',v'))=k_{\\text{pos}}(x,x')k_{\\text{dir}}(v,v')` where
    :math:`k_{\\text{pos}}((t,x),(t',x'))=K^{(1)}_{\\sigma_{\\text{pos},t}}(t,t')K^{(d)}_{\\sigma_{\\text{pos},x}}(x,x')`, 
    :math:`k_{\\text{dir}}((t,x),(t',x'))=K^{(1)}_{\\sigma_{\\text{dir},t}}(t,t')K^{(d)}_{\\sigma_{\\text{dir},x}}(x,x')`
    and 
    :math:`K_{\\sigma}^{(a)}(w,w')=\\exp(-|w-w'|^2/\\sigma))` the :math:`a` dimensional isotropic Gaussian kernel of variance :math:`\\sigma`.

    Parameters
    ----------
        t_sigma_1 : (float)
             variance of the kernel related to time coordinate of the kernel related to position :math:`\\sigma_{\\text{pos},t}`

        s_sigma_1 : (float)
             variance of the kernel related to space coordinate of the kernel related to position :math:`\\sigma_{\\text{pos},x}`

        t_sigma_2 : (float)
             variance of the kernel related to time coordinate of the kernel related to direction :math:`\\sigma_{\\text{dir},t}`

        s_sigma_2 : (float)
             variance of the kernel related to space coordinate of the kernel related to direction :math:`\\sigma_{\\text{dir},x}`

        np_dtype : (_type_, optional):
             Defaults to np.float32.

    Returns
    -------
        kernel function : (X,mask_X,Y,mask_Y,b)-> array of the shape of b
            :math:`K(X,Y)b` where X and Y are array of size (n_samples,d+1), :math:`K(X,Y)` is the kernel matrix :math:`(k(x_i,y_j))`
            and b is an array of shape (n_samples,d) with d the dimension of the problem

    """
    # important for the varifold loss 
    t_oos_1, s_oos_1  = 1/t_sigma_1, 1/s_sigma_1
    t_oos_2, s_oos_2 = 1/t_sigma_2, 1/s_sigma_2
    def K(x,y,u,v,mask_xu,mask_yv,b): 
        n_d = x.shape[1]
        oos_1 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_1[0,0], oos_1[0,1:] = t_oos_1, s_oos_1
        k1 = jnp.exp(-jnp.sum(((x*oos_1)[:,None,:]-(y*oos_1)[None,:,:])**2,axis=2))
        oos_2 = np.ones(n_d,dtype=np_dtype).reshape(1,-1)
        oos_2[0,0], oos_2[0,1:] = t_oos_2,s_oos_2
        k2 = jnp.exp(-jnp.sum(((u*oos_2)[:,None,:]-(v*oos_2)[None,:,:])**2,axis=2))
        mask = (mask_xu*mask_yv.T)
        return (k1*k2*mask)@b
    return K
def Velocity_TSLDDMM_Gaussian_Kernel(c_0:float,c_1:float,t_sigma_1:float,t_sigma_2:float,s_sigma:float): 
    """
    Function related to the velocity TSLDDMM gaussian kernel defined as,  using the notations of the paper "Shape analysis for time series" :math:`K_G((t,x),(t',x'))=\\begin{pmatrix} c_0 K_{\\text{time}}(t,t') & 0 \\\\ 0& c_1 K_{\\text{space}}((t,x),(t',x'))\\end{pmatrix}`
    with :math:`K_{\\text{time}}=K_{\\sigma_{T,0}}^{(1)}(t,t'), \quad K_{\\text{space}}=K_{\\sigma_{T,1}}^{(1)}(t,t')K_{\\sigma_{x}}^{(d)}(x,x')`
    and :math:`K_{\\sigma}^{(a)}(w,w')=\\exp(-|w-w'|^2/\\sigma))` the :math:`a` dimensional isotropic Gaussian kernel of variance :math:`\\sigma`.

    Parameters
    ----------

        c_0 : (float)
            time scaling :math:`c_0`

        c_1 : (float)
            space scaling :math:`c_1`

        t_sigma_1 : (float)
            the time variance parameter 1 :math:`\\sigma_{T,0}`

        t_sigma_2 : (float)
            the time variance parameter 2 :math:`\\sigma_{T,1}`

        s_sigma : (float)
            the space variance parameter :math:`\\sigma_{x}`

    Returns
    -------

        kernel function : (X,mask_X,Y,mask_Y,b)-> array of the shape of b
            :math:`K_G(X,Y)b` where X and Y are array of size (n_samples,d+1), :math:`K(X,Y)` is the kernel matrix :math:`(k(x_i,y_j))`
            and b is an array of shape (d,) with d the dimension of the problem

    """
    #variance parameter
    t_oos_12 = 1/t_sigma_1**2
    t_oos_22 = 1/t_sigma_2**2
    s_oos2 = 1/s_sigma**2
    def K(x,mask_x,y,mask_y,b): 
        #return K_X,Y@b
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_b, s_b = b[:,:1], b[:,1:]
        #time difference, for the kernel related to time
        time_sum = jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)
        t_res_1 = jnp.exp(-t_oos_12*time_sum)
        # value difference, for the kernel related to space
        s_res = jnp.exp(-s_oos2*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)-t_oos_22*time_sum)
        mask = mask_x*mask_y.T
        #aggregation (mu*K_time@t_b,lambda*K_space@s_b)
        return jnp.hstack((c_0*(mask*t_res_1)@(t_b*mask_y), c_1*(mask*s_res)@(s_b*mask_y)))
    return K 



def Velocity_TSLDDMM_Cauchy_Kernel(c_0:float,c_1:float,t_sigma_1:float,t_sigma_2:float,s_sigma:float): 
    """
    Function related to the velocity TSLDDMM cauchy kernel defined as, using the notations of the paper "Shape analysis for time series"
    :math:`K_G((t,x),(t',x'))=\\begin{pmatrix} c_0 K_{\\text{time}}(t,t') & 0 \\\\ 0& c_1 K_{\\text{space}}((t,x),(t',x'))\\end{pmatrix}`
    with :math:`K_{\\text{time}}=K_{\\sigma_{T,0}}^{(1)}(t,t'), \quad K_{\\text{space}}=K_{\\sigma_{T,1}}^{(1)}(t,t')K_{\\sigma_{x}}^{(d)}(x,x')`
    and :math:`K_{\\sigma}^{(a)}(w,w')=\\frac{1}{1+|w-w'|^2/\\sigma)}` the :math:`a` dimensional isotropic Cauchy kernel of variance :math:`\\sigma`.

    Parameters
    ----------
        c_0 : (float)
            time scaling :math:`c_0`

        c_1 : (float)
            space scaling :math:`c_1`

        t_sigma_1 : (float)
            the time variance parameter 1 :math:`\\sigma_{T,0}`

        t_sigma_2 : (float)
            the time variance parameter 2 :math:`\\sigma_{T,1}`

        s_sigma : (float)
            the space variance parameter :math:`\\sigma_{x}`

    Returns
    -------
    
        kernel function : (X,mask_X,Y,mask_Y,b)-> array of the shape of b
            :math:`K_G(X,Y)b` where X and Y are array of size (n_samples,d+1), :math:`K(X,Y)` is the kernel matrix :math:`(k(x_i,y_j))`
            and b is an array of shape (d,) with d the dimension of the problem


    """
    t_oos_12 = 1/t_sigma_1**2
    t_oos_22 = 1/t_sigma_2**2
    s_oos2 = 1/s_sigma**2
    def K(x,mask_x,y,mask_y,b): 
        t_x, s_x = x[:,:1], x[:,1:]
        t_y, s_y = y[:,:1], y[:,1:]
        t_b, s_b = b[:,:1], b[:,1:]
        time_sum = jnp.sum((t_x[:,None,:]-t_y[None,:,:])**2,axis=2)
        t_res_1 = 1/(1+t_oos_12*time_sum)
        s_res = 1/(1+s_oos2*jnp.sum((s_x[:,None,:]-s_y[None,:,:])**2,axis=2)+t_oos_22*time_sum)
        mask = mask_x*mask_y.T
        return jnp.hstack((c_0*(mask*t_res_1)@(t_b*mask_y), c_1*(mask*s_res)@(s_b*mask_y)))
    return K 





