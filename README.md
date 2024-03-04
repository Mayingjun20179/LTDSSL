# LTDSSL

This is the data and code for the paper "Yingjun Ma, Junjiang Zhong. Logistic tensor decomposition with sparse subspace learning for prediction of multiple disease types of human–virus protein–protein interactions[J]. Briefings in Bioinformatics, 2023, 24(1), bbac604. DOI: 10.1093/bib/bbac604".  Please cite if you use this code.
(Written by Yingjun Ma, 2023)

This is the data and code for the paper "Logistic tensor decomposition with sparse subspace learning for prediction of multiple disease types of human-virus protein-protein interactions".  Please cite if you use this code.

Data description:

"exper_data.mat" is the data format of matlab, which stores all the data of this experiment

"Experimental_data_all.xlsx" is the data file of this study, including human (virus) protein ID, human (virus) protein sequence, human (virus) protein pseAAC feature, human (virus) protein CTD feature, and 6 Human (viral) protein association in central vascular disease.



Code description:

"LTDSSL_opt.m" code for the LTDSSL method of this study

"mianZ.m" is the main function, and running this function can get the results of "CV_type" , "CV_triplet",'cv_human',and 'cv_virus'.

In this package, we used the tensor tensor_toolbox-v3.1, which is downloaded from (https://gitlab.com/tensors/tensor_toolbox/-/releases/v3.1)
