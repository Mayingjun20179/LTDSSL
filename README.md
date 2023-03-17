# LTDSSL
Logistic tensor decomposition with sparse subspace learning for prediction of multiple disease types of human-virus protein-protein interactions

This is the data and code for the paper "Logistic tensor decomposition with sparse subspace learning for prediction of multiple disease types of human-virus protein-protein interactions".  Please cite if you use this code.

Data description:
"exper_data.mat" is the data format of matlab, which stores all the data of this experiment
"Experimental_data_all.xlsx" is the data file of this study, including human (virus) protein ID, human (virus) protein sequence, human (virus) protein pseAAC feature, human (virus) protein CTD feature, and 6 Human (viral) protein association in central vascular disease.

Code description:
"LTDSSL_opt.m" code for the LTDSSL method of this study
"mianZ.m" is the main function, and running this function can get the results of "CV_type" , "CV_triplet",'cv_human',and 'cv_virus'.

In addition, you need to download "tensor_toolbox-v3.1" before running, and place it in the running path of matlab.
