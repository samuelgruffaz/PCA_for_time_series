import numpy as np
from .kernel import Varifold_TSLDDMM_Gaussian_Kernel,Velocity_TSLDDMM_Gaussian_Kernel
from .lddmm import Shooting, batch_one_to_many_registration
from .utils import time_shape_embedding,batch_dataset
from .loss import SumVarifoldLoss
from .statistic import MomentaPCA
from .barycenter import batch_barycenter_registration
from optax.schedules import warmup_cosine_decay_schedule
import optax
import matplotlib.pyplot as plt

#TODO preprocessing, from_timeseries_to_dataset, tuto, synthetic




#kernel_velocity
#c_{time}->c_0
#c_{space}->c_1
#\sigma_{time,0}->t_sigma_1
#\sigma_{time,1}->t_sigma_2
#\sigma_{space,1}->s_sigma

#kernel varifold
# "$\sigma_{pos,t}$", min_value =0.1, max_value =4.0, step=0.1,value=2.0)->t_sigma_1
# "$\sigma_{pos,x}$", min_value =0.1, max_value =4.0, step=0.1,value=1.0)->s_sigma_1
# $\sigma_{dir,t}$", min_value =0.1, max_value =4.0, step=0.1,value=2.0)->t_sigma_2
# $\sigma_{dir,x}$", min_value =0.1, max_value =4.0,step= 0.1,value=0.6)->s_sigma_2

Defaut_velocity_hyper={"c_0":1.0,"c_1":0.1,"t_sigma_1":60,"t_sigma_2":1.0,"s_sigma":1.0}
Defaut_varifold_hyper={"t_sigma_1":2.0,"s_sigma_1":1.0,"t_sigma_2":2.0,"s_sigma_2":0.6}

niter_defaut=100
schedule_defaut =warmup_cosine_decay_schedule(0,0.3,int(0.1*niter_defaut),niter_defaut,0)
#schedule =warmup_cosine_decay_schedule(0,0.3,80,800,0)


class TS_PCA_():

    def __init__(self,kernel_velocity_arg=Defaut_velocity_hyper,kernel_varifold_arg=Defaut_varifold_hyper):
        """
        Initialize the TS-PCA object with velocity and varifold kernels for TS-LDDMM.

        The velocity kernel is a Gaussian kernel defined as in "Shape analysis for time series":
        :math:`K_G((t,x),(t',x')) = \begin{pmatrix} c_0 K_{\text{time}}(t,t') & 0 \\ 0 & c_1 K_{\text{space}}((t,x),(t',x')) \end{pmatrix}`,
        where :math:`K_{\text{time}}` and :math:`K_{\text{space}}` are isotropic Gaussian kernels with parameters specified in `kernel_velocity_arg`.

        The varifold kernel is a Gaussian kernel used to compute distances between shapes along the time series,
        with parameters specified in `kernel_varifold_arg`.

        Parameters
        ----------
            kernel_velocity_arg : dict, optional
                Hyperparameters for the velocity kernel (default: `Defaut_velocity_hyper`):
                - `c_0` : float
                    Time scaling coefficient.
                - `c_1` : float
                    Space scaling coefficient.
                - `t_sigma_1` : float
                    Variance parameter for the temporal component (sigma T0).
                - `t_sigma_2` : float
                    Variance parameter for the temporal component (sigma T1).
                - `s_sigma` : float
                    Variance parameter for the spatial component (sigma_x).

            kernel_varifold_arg : dict, optional
                Hyperparameters for the varifold kernel (default: `Defaut_varifold_hyper`):
                - `t_sigma_1`, `t_sigma_2` : float
                    Variance parameters for the temporal components.
                - `s_sigma_1`, `s_sigma_2` : float
                    Variance parameters for the spatial components.

        Attributes
        ----------
            velocity_kernel : Velocity_TSLDDMM_Gaussian_Kernel
                Gaussian kernel used for the velocity in TS-LDDMM.
            varifold_kernel : Varifold_TSLDDMM_Gaussian_Kernel
                Gaussian kernel used for computing varifold distances.
            shoot : Shooting
                Shooting object initialized with the velocity kernel for TS-LDDMM computations.
            fit_TS_LDDMM : bool
                Flag indicating whether TS-LDDMM representations have been fitted (initialized as False).

        """
        
        
        self.velocity_kernel=Velocity_TSLDDMM_Gaussian_Kernel(kernel_velocity_arg["c_0"],kernel_velocity_arg["c_1"],kernel_velocity_arg["t_sigma_1"],kernel_velocity_arg["t_sigma_2"],kernel_velocity_arg["s_sigma"])
        self.varifold_kernel=Varifold_TSLDDMM_Gaussian_Kernel(kernel_varifold_arg["t_sigma_1"],kernel_varifold_arg["s_sigma_1"],kernel_varifold_arg["t_sigma_2"],kernel_varifold_arg["s_sigma_2"])
        self.shoot= Shooting(self.velocity_kernel)
        self.fit_TS_LDDMM=False


    def fit_TS_LDDMM_representations(self,Dataset,Dataset_mask,niter=niter_defaut,schedule=schedule_defaut,gamma_loss=1e-3,learning_graph_ref=True,graph_ref=None,graph_ref_mask=None,dataloss=None):
        """
        Fit TS-LDDMM representations of a dataset of time series graphs.

        This method computes the TS-LDDMM embeddings of the input dataset by optimizing 
        initial momentums (:math:`p_0`) and optionally the reference graph 
        (:math:`G_0`) using a velocity kernel and a varifold data term. 
        If `learning_graph_ref` is True, the reference graph is estimated via a barycenter 
        procedure; otherwise, a fixed reference graph must be provided.

        Parameters
        ----------
            Dataset : array of shape (n_time_series, n_samples_max, d+1)
                Input time series graphs. The first column corresponds to time points, 
                the remaining columns are spatial values of dimension d.

            Dataset_mask : array of shape (n_time_series, n_samples_max, 1)
                Mask indicating valid samples in each time series graph.

            niter : int, optional
                Number of iterations for the optimization (default: `niter_defaut`).

            schedule : object, optional
                Learning rate schedule for the optax optimizer (default: `schedule_defaut`).

            gamma_loss : float, optional
                Weighting factor for the data loss term in the optimization (default: 1e-3).

            learning_graph_ref : bool, optional
                If True, the reference graph `graph_ref` is estimated jointly with momentums;
                if False, a fixed reference graph must be provided (default: True).

            graph_ref : array of shape (n_samples_max, d+1), optional
                Reference graph to use if `learning_graph_ref=False`. Must be provided together 
                with `graph_ref_mask`.

            graph_ref_mask : array of shape (n_samples_max, 1), optional
                Mask for the reference graph.

            dataloss : object, optional
                Data loss object used for registration. If None, a default `SumVarifoldLoss` 
                with the internal varifold kernel is used.

        Returns
        -------
            momentums : array
                Optimized initial momentums (:math:`p_0`) for the TS-LDDMM mapping.

            graph_ref : array
                Learned or provided reference graph (:math:`G_0`) after optimization.

            graph_ref_mask : array
                Mask for the reference graph.

        Attributes
        ----------
            fit_TS_LDDMM_bool : bool
                Flag indicating that TS-LDDMM representations have been fitted successfully 
                (set to True at the end of the method).

        Notes
        -----
        - This function internally uses batching via `batch_dataset` and the `Shooting` object.
        - Optimization is performed using the Adabelief optimizer from `optax`.
        - Raises a ValueError if `learning_graph_ref=False` and `graph_ref` or `graph_ref_mask` are not provided.
        """
        bDataset,bDataset_mask = batch_dataset(Dataset,1,Dataset_mask)
        optimize=optax.adabelief(schedule)

        if dataloss is None:
            dataloss=SumVarifoldLoss([self.varifold_kernel])
        if not(learning_graph_ref) :
            if graph_ref is None or graph_ref_mask is None:
                raise ValueError("You should give a graph of reference 'graph_ref' array(n_samples,d+1) and its mask 'graph_ref_mask' array(n_samples,1)")
            self.graph_ref=graph_ref# G_0
            self.graph_ref_mask=graph_ref_mask
            print("run momentums learning")
            p0s,q0,q0_mask= batch_one_to_many_registration(self.graph_ref,self.graph_ref_mask,bDataset,bDataset_mask,self.velocity_kernel,dataloss,niter=niter,optimizer=optimize,gamma_loss=gamma_loss,stream_bool=False)
            self.momentums=p0s
        else:
            print("run momentums+graph_ref learning")
            p0s,q0,q0_mask = batch_barycenter_registration(self.shoot,bDataset,bDataset_mask,self.velocity_kernel,dataloss,niter=niter,optimizer=optimize,gamma_loss=gamma_loss,stream_bool=False)
            self.momentums=p0s
            self.graph_ref=q0# G_0
            self.graph_ref_mask=q0_mask
        self.fit_TS_LDDMM_bool=True
        return self.momentums,self.graph_ref,self.graph_ref_mask



    def fit_kernel_PCA(self,n_comp=3,momentums=None,graph_ref=None,graph_ref_mask=None):
        """
        Fit a Kernel PCA on TS-LDDMM momentums to extract principal components.

        This method performs Kernel Principal Component Analysis (KPCA) on the initial 
        momentums (:math:`p_0`) obtained from TS-LDDMM representations. It computes 
        a low-dimensional embedding of the momentums, capturing the main modes of 
        variation in the dataset.

        If `momentums` are not provided, the method uses the momentums, reference graph,
        and reference graph mask stored in the object (from a previous call to 
        `fit_TS_LDDMM_representations`).

        Parameters
        ----------
            n_comp : int, optional
                Number of principal components to extract (default: 3).

            momentums : array of shape (n_ind, n_samples, n_dim+1), optional
                TS-LDDMM initial momentums (:math:`p_0`). If None, the method uses the 
                internal momentums stored in the object.

            graph_ref : array of shape (n_samples, d+1), optional
                Reference graph (:math:`G_0`) corresponding to the momentums. Required if 
                `momentums` is provided.

            graph_ref_mask : array of shape (n_samples, 1), optional
                Mask for the reference graph. Required if `momentums` is provided.

        Returns
        -------
            None

        Attributes
        ----------
            n_comp : int
                Number of components used for KPCA.

            mean_ : array
                Mean momentum (:math:`\\bar{p}_0`) computed across samples.

            singular_values_ : array
                Standard deviation of principal components.

            components_ : array
                Principal components matrix, each row corresponds to a component.

            n_samples_ : int
                Number of time series in the dataset.

            coordinates_ : array
                Projected coordinates of each momentum in the KPCA space.

            fit_kernel_PCA_bool : bool
                Flag indicating that Kernel PCA has been successfully fitted (set to True).

        Raises
        ------
            ValueError
                If `momentums`, `graph_ref`, or `graph_ref_mask` are not provided and 
                cannot be inferred from the object. Suggests using 
                `fit_TS_LDDMM_representations` to generate these inputs.

        Notes
        -----

            - The method internally uses `MomentaPCA` with covariance-based KPCA.
            - Each momentum is reshaped to match the expected input dimensions for KPCA.
            - Kernel PCA captures the main modes of variation of TS-LDDMM momentums,
            which can then be used for visualization or downstream analysis.

        """
        if self.fit_TS_LDDMM_bool and momentums is None:
            momentums,graph_ref,graph_ref_mask=self.momentums,self.graph_ref,self.graph_ref_mask
            momentums = momentums.reshape(-1,momentums.shape[2],momentums.shape[3])
        
        if momentums is None or graph_ref is None or graph_ref_mask is None:
            raise ValueError("You should give momentums (n_ind,n_samples,n_dim+1) , a graph of reference 'graph_ref' array(n_samples,d+1) and its mask 'graph_ref_mask' array(n_samples,1). You should use 'fit_TS_LDDMM_representations' to get it. ")
        else:
            print("run kernel PCA on the momentums")
            mpca = MomentaPCA(n_comp,False,"cov")
            
            mpca.fit(self.velocity_kernel,momentums,graph_ref,graph_ref_mask)
        self.n_comp=n_comp
        self.mean_ = mpca.m_ps_ #mean p0
        self.singular_values_=mpca.p_std_
        self.components_=mpca.p_pc_ # each line
        self.n_samples_=len(momentums)
        self.coordinates_=mpca.p_score_
        self.fit_kernel_PCA_bool=True


    def fit_transform(self,n_comp=2,momentums=None,graph_ref=None,graph_ref_mask=None):
        """

        Apply fit_kernel_PCA and return the Projected coordinates of each momentum in the Kernel PCA space.

        """
        self.fit_kernel_PCA(self,n_comp=n_comp,momentums=momentums,graph_ref=graph_ref,graph_ref_mask=graph_ref_mask)
        return self.coordinates_
    
    def plot_components(self):
        """

        Plot the the effect of the components of Kernel PCA as deformations on the reference graph

        """

        if not(self.fit_kernel_PCA_bool):
            raise ValueError(" You should first call 'fit_kernel_PCA' to get the components. ")

        else:
            n_disp=3
            ft_size = 25
            pft_size = 20
            components = [f"PC{i+1}" for i in range(self.n_comp)]
            p0_b = self.mean_
            plot_mask = np.where(self.graph_ref_mask==True)[0]
            fig2,axs = plt.subplots(self.n_comp,n_disp,figsize = (n_disp*3,self.n_comp*3),sharex=True,sharey=True)
            for pca_index in range(self.n_comp):
                sigma_pca = self.singular_values_[pca_index]
                for j,alpha in enumerate(np.linspace(-2*sigma_pca , 2*sigma_pca, n_disp)):
                    p0_mode = p0_b + alpha * self.components_[pca_index]
                    _,q = self.shoot(p0_mode,self.graph_ref,self.graph_ref_mask)
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

            plt.title("kernelPCA components related to the TS-LDDMM representation")
            plt.show()