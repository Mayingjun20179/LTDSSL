%%%%%%%%%%%%%LTDSSL,CP,BCPFARD,5-fold
clc
clear

%%%Import experimental data
load('exper_data.mat')

%%%%%%%%%%%%%%%%%%%%%%%%LTDSSL
method_name = 'LTDSSL';
main_cv(exper_data,method_name,'cv_type');    %%%%CV_type
main_cv(exper_data,method_name,'cv_triplet');   %%%%CV_triplet
main_cv(exper_data,method_name,'cv_human');   %%%%CV_human
main_cv(exper_data,method_name,'cv_virus');   %%%%CV_virus

