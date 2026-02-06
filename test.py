from TS_PCA import Varifold_TSLDDMM_Gaussian_Kernel,TS_PCA_,generate_easy_dataset


N=5
dataset,dataset_mask,graph_ref,graph_ref_mask=generate_easy_dataset(N=8)

class_test=TS_PCA_()

class_test.fit_TS_LDDMM_representations(dataset,dataset_mask,learning_graph_ref=False,graph_ref=graph_ref,graph_ref_mask=graph_ref_mask)

class_test.fit_kernel_PCA()

class_test.plot_components()