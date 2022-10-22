 %%%%%%%%%%%%%
clc
clear
%%%%cv_type
load('exper_data.mat')
flag = 1;   %cv_type
main_cv(exper_data,flag);

%%%%cv_triplet
flag = 2;   %cv_triplet
main_cv(exper_data,flag);



