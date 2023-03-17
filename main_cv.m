function main_cv(DATA,method_name,cv_exper)
inter_tensor = DATA.inter_tensor_exp;
% paramater

k_folds = 5;

%%%LTDSSL
if strcmp(method_name,'LTDSSL')==1   
    option = [4,1,4,6,1];
    if strcmp(cv_exper,'cv_type')==1        
        cv_data = cross_validation1(inter_tensor,k_folds);
        result = cv_LTDSSL_type(DATA,cv_data,option);
        LTDSSL_type = result;
        save .\LTDSSL_type LTDSSL_type;
    elseif strcmp(cv_exper,'cv_triplet')==1
        cv_data = cross_validation2(inter_tensor,k_folds);
        result = cv_LTDSSL_triplet(DATA,cv_data,option);
        LTDSSL_triplet = result;
        save .\LTDSSL_triplet LTDSSL_triplet;
    elseif strcmp(cv_exper,'cv_human')==1
        cv_data = cross_validation3(inter_tensor,k_folds);
        result = cv_LTDSSL_human(DATA,cv_data,option);
        LTDSSL_human = result; 
        save .\LTDSSL_human LTDSSL_human;
    elseif strcmp(cv_exper,'cv_virus')==1
        cv_data = cross_validation4(inter_tensor,k_folds);
        result = cv_LTDSSL_virus(DATA,cv_data,option);
        LTDSSL_virus = result;
        save .\LTDSSL_virus LTDSSL_virus;
    end
end

end



%%%perform LTDSSL cv_type
function result = cv_LTDSSL_type(DATA,cv_data,option)
k_folds = length(cv_data);
result = 0;
FH = {DATA.human_paac_fea,DATA.human_ctd_fea};
FV = {DATA.virus_paac_fea,DATA.virus_ctd_fea};
FHV = {FH,FV};
real_tensor = DATA.inter_tensor_exp;
SHV = {DATA.human_paac_sim,DATA.virus_paac_sim};

for k=1:k_folds
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%
    train_tensor = cv_data{k,1};    %train matrix
    %%%%%%
    pre_tensor = LTDSSL_opt(train_tensor,FHV,option);
    pre_tensor = double(pre_tensor);
    %    
    resultk =  evaluate_opt1(pre_tensor,real_tensor,cv_data{k,2});    
    result = result + resultk;
    result/k
    toc;
end
result = result/k_folds;
end
%%%perform LTDSSL cv_triplet
function result = cv_LTDSSL_triplet(DATA,cv_data,option)
cv = length(cv_data);
result = 0;
FH = {DATA.human_paac_fea,DATA.human_ctd_fea};
FV = {DATA.virus_paac_fea,DATA.virus_ctd_fea};
FHV = {FH,FV};

for k=1:cv
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%
    train_tensor = cv_data{k,1};    %train matrix
    %%%%%
    pre_tensor = LTDSSL_opt(train_tensor,FHV,option);
    pre_tensor = double(pre_tensor);
    %%%%%
    N_pos = length(cv_data{k,2});
    resultk = [];
    seeds = 1:1000:20000;
    for i=1:length(seeds)
        ind_neg = find(DATA.inter_tensor_exp==0);
        rand('state',seeds(i));
        index = randperm(length(ind_neg));
        ind_neg = ind_neg(index(1:N_pos));
        ind_test = [cv_data{k,2};ind_neg];
        label = [ones(N_pos,1);zeros(N_pos,1)];
        resultk = [resultk;evaluate_opt2(pre_tensor(ind_test),label)];
    end
    result = result + mean(resultk);   
    result/k
    toc;      
end
result = result/cv;
end
%%%perform LTDSSL cv_human
function result = cv_LTDSSL_human(DATA,cv_data,option)
cv = length(cv_data);
result = 0;
FH = {DATA.human_paac_fea,DATA.human_ctd_fea};
FV = {DATA.virus_paac_fea,DATA.virus_ctd_fea};
FHV = {FH,FV};
SHV = {DATA.human_paac_sim,DATA.virus_paac_sim};
for k=1:cv
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%
    train_tensor = cv_data{k,1};    %train matrix
    train_tensor = xiuzheng_Y(train_tensor,SHV);   
    %%%%%
    pre_tensor = LTDSSL_opt(train_tensor,FHV,option);
    pre_tensor = double(pre_tensor);
    %%%%%
    test_row_ind = cv_data{k,2};    
    label_row = DATA.inter_tensor_exp(test_row_ind,:,:);   %%%Position of all rows
    pre_row = pre_tensor(test_row_ind,:,:);
    ind_pos = find(label_row==1);  
    N_pos = length(ind_pos);   %Number of positive cases
    resultk = [];
    seeds = 1:1000:20000;
    for i=1:length(seeds)        
        ind_neg = find(label_row==0);
        rand('state',seeds(i));
        index = randperm(length(ind_neg));
        ind_neg = ind_neg(index(1:N_pos));
        ind_test = [ind_pos;ind_neg];
        label = [ones(N_pos,1);zeros(N_pos,1)];
        resultk = [resultk;evaluate_opt2(pre_row(ind_test),label)];
    end
    result = result + mean(resultk);  
    result/k
    toc;      
end
result = result/cv;
end

%%%perform LTDSSL cv_virus
function result = cv_LTDSSL_virus(DATA,cv_data,option)
cv = length(cv_data);
result = 0;
FH = {DATA.human_paac_fea,DATA.human_ctd_fea};
FV = {DATA.virus_paac_fea,DATA.virus_ctd_fea};
FHV = {FH,FV};
SHV = {DATA.human_paac_sim,DATA.virus_paac_sim};
for k=1:cv
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%
    train_tensor = cv_data{k,1};    %train matrix
    train_tensor = xiuzheng_Y(train_tensor,SHV); 
    %%%%%
    pre_tensor = LTDSSL_opt(train_tensor,FHV,option);
    pre_tensor = double(pre_tensor);
    %%%%%
    test_col_ind = cv_data{k,2};
    label_col = DATA.inter_tensor_exp(:,test_col_ind,:);   %%%Position of all cols
    pre_col = pre_tensor(:,test_col_ind,:);
    ind_pos = find(label_col==1);  
    N_pos = length(ind_pos);   %Number of positive cases
    resultk = [];
    seeds = 1:1000:20000;
    for i=1:length(seeds)        
        ind_neg = find(label_col==0);
        rand('state',seeds(i));
        index = randperm(length(ind_neg));
        ind_neg = ind_neg(index(1:N_pos));
        ind_test = [ind_pos;ind_neg];
        label = [ones(N_pos,1);zeros(N_pos,1)];
        resultk = [resultk;evaluate_opt2(pre_col(ind_test),label)];
    end
    result = result + mean(resultk);  
    result/k
    toc;      
end
result = result/cv;   
end


%intMat: interaction matrix
%k_folds=5,five fold cross CV type
function cv_data = cross_validation1(inter_tensor,k_folds)

association_matrix = sum(inter_tensor,3);
[ind_x,ind_y] = find(association_matrix>0);
index_matrix = [ind_x,ind_y];
pair_num = size(index_matrix,1);
sample_num_per_fold = floor(pair_num / k_folds);


cv_data = cell(k_folds,2);  
rand('state',1000);
index = randperm(pair_num)';
index_matrix = index_matrix(index,:);


for j = 1:k_folds   %
    if j < k_folds
        test_pos_ind = index_matrix((j-1)*sample_num_per_fold+1:j*sample_num_per_fold,:);
    else
        test_pos_ind = index_matrix(((j-1)*sample_num_per_fold+1):end,:);
    end
    train_tensor = inter_tensor;
    train_tensor = test_change(test_pos_ind,train_tensor);
    cv_data(j,:) = {train_tensor, test_pos_ind};
end

end

%intMat: interaction matrix
%cv=5,five fold cross CV triplet
function cv_data = cross_validation2(inter_tensor,cv)
cv_data = cell(cv,2);  
pos_indx = find(inter_tensor==1);
num_pos = length(pos_indx);
% if exist('seed','var') == 1
%     rand('state',seed)
% end
rand('state',1);
index = randperm(num_pos)';
pos_indx = pos_indx(index);
num_step =  floor(num_pos/cv);

for j = 1:cv   %
    if j < cv
        test_pos_ind = pos_indx((j-1)*num_step+1:j*num_step);
    else
        test_pos_ind = pos_indx(((j-1)*num_step+1):end);
    end
    train_tensor = inter_tensor;
    train_tensor(test_pos_ind) = 0;
    cv_data(j,:) = {train_tensor, test_pos_ind};
end

end


%intMat: interaction matrix
%cv=5,five fold cross CV human protein
function cv_data = cross_validation3(inter_tensor,cv)
cv_data = cell(cv,2);  
[Nh,~,~] = size(inter_tensor);
rand('state',1);
index = randperm(Nh)';
num_step =  floor(Nh/cv);
for j = 1:cv   %    
    if j < cv
        test_row_ind = index((j-1)*num_step+1:j*num_step);         
    else
        test_row_ind = index(((j-1)*num_step+1):end);
    end
    train_tensor = inter_tensor;
    train_tensor(test_row_ind,:,:) = 0;
    cv_data(j,:) = {train_tensor, test_row_ind};
end

end


% intMat: interaction matrix
% cv=5,five fold cross CV virus protein
function cv_data = cross_validation4(inter_tensor,cv)
cv_data = cell(cv,2);  
[~,Nv,~] = size(inter_tensor);
rand('state',1);
index = randperm(Nv)';
num_step =  floor(Nv/cv);
for j = 1:cv   %    
    if j < cv
        test_col_ind = index((j-1)*num_step+1:j*num_step);        
    else
        test_col_ind = index(((j-1)*num_step+1):end);        
    end
    train_tensor = inter_tensor;
    train_tensor(:,test_col_ind,:) = 0;
    cv_data(j,:) = {train_tensor, test_col_ind};
end

end



%%%
function train_tensor = test_change(test_pos_ind,train_tensor)
N3 = size(train_tensor,3);
N = size(test_pos_ind,1);
for  ii = 1:N3
    for jj = 1:N
        train_tensor(test_pos_ind(jj,1),test_pos_ind(jj,2),ii)=0;
    end
end
    
    
end
  
%%% Evaluate cv_type
function resultk =  evaluate_opt1(pre_tensor,real_tensor,pos_ind)

%pre_tensor  
%real_tensor 
%pos_ind
TP = 0;
real_sum = 0;
recall = 0;
sample_num_eval = size(pos_ind,1);
Ntype = size(pre_tensor,3);
for i=1:sample_num_eval
    ind_x = pos_ind(i,1);
    ind_y = pos_ind(i,2);
    %%
    predict_score = [];
    for j = 1:Ntype
        predict_score = [predict_score,pre_tensor(ind_x,ind_y,j)];
    end
    %%
    real_score = [];
    for j = 1:Ntype
        real_score = [real_score,real_tensor(ind_x,ind_y,j)];
    end
    
    positive_num = sum(real_score);  %
    real_sum = real_sum + positive_num;
    
    [~,sort_index] = sort(predict_score,'ascend');
    predict_score(find(predict_score~=0))=0;
    predict_score(sort_index(end)) = 1;
    tp = predict_score * real_score';
    if tp>2
        disp(i)
    end
    TP = TP + tp;
    recall = recall + tp / positive_num;
end
avg_precision = TP / sample_num_eval;
mi_avg_recall = TP / real_sum;
ma_avg_recall = recall / sample_num_eval;
resultk = [avg_precision,mi_avg_recall,ma_avg_recall];
end

%%% Evaluate cv_triplet
function result = evaluate_opt2(score,label)
%%%%TN,TP,FN,FP
sort_predict_score=unique(sort(score));
score_num = length(sort_predict_score);
Nstep = 2000;
threshold=sort_predict_score(ceil(score_num*(1:Nstep)/(Nstep+1)));

threshold=threshold';
threshold = threshold(end:-1:1);
threshold_num=length(threshold);
TN=zeros(threshold_num,1);
TP=zeros(threshold_num,1);
FN=zeros(threshold_num,1);
FP=zeros(threshold_num,1);

for i=1:threshold_num
    tp_index=logical(score>=threshold(i) & label==1);
    TP(i,1)=sum(tp_index);
    
    tn_index=logical(score<threshold(i) & label==0);
    TN(i,1)=sum(tn_index);
    
    fp_index=logical(score>=threshold(i) & label==0);
    FP(i,1)=sum(fp_index);
    
    fn_index=logical(score<threshold(i) & label==1);
    FN(i,1)=sum(fn_index);
end


%%%%%计算AUPR
SEN=TP./(TP+FN);
PRECISION=TP./(TP+FP);
recall=SEN;
x=recall;
y=PRECISION;
[x,index]=sort(x);
y=y(index,:);
x = [0;x];  y = [1;y];
x(end+1,1)=1;  y(end+1,1)=0;
AUPR=0.5*x(1)*(1+y(1));
for i=1:threshold_num
    AUPR=AUPR+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end


AUPR_xy = [x,y];

%%%%%计算AUC
AUC_x = FP./(TN+FP);      %FPR
AUC_y = TP./(TP+FN);      %tpR
[AUC_x,ind] = sort(AUC_x);
AUC_y = AUC_y(ind);
AUC_x = [0;AUC_x];
AUC_y = [0;AUC_y];
AUC_x = [AUC_x;1];
AUC_y = [AUC_y;1];

AUC = 0.5*AUC_x(1)*AUC_y(1)+sum((AUC_x(2:end)-AUC_x(1:end-1)).*(AUC_y(2:end)+AUC_y(1:end-1))/2);

AUCxy = [AUC_x(:),AUC_y(:)];

%%%%%
temp_accuracy=(TN+TP)/length(label);   
temp_sen=TP./(TP+FN);    
recall = temp_sen;
temp_spec=TN./(TN+FP);   
temp_precision=TP./(TP+FP); 
temp_f1=2*recall.*temp_precision./(recall+temp_precision);
[~,index]=max(temp_f1);
%%%
f1=temp_f1(index);
result=[AUPR,AUC,f1];
end




