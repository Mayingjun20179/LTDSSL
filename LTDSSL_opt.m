%%%%逻辑张量分解+稀疏子空间学习
function pre_tensor = LTDSSL_opt(train_tensor, FUV,option)
R = option(1);  %表示因子矩阵的维度
ar = option(2);  %特征正则化参数
c = option(3);
yita = option(4);
miu = option(5);

%初始化投影系数PUV
PUV  = cell(1,2);
for i=1:2
    Ni = length(FUV{i});
    Pi = cell(1,Ni);
    for j = 1:Ni        
        M = size(FUV{i}{j},2);
        rand('state',1);
        Pi{j} = rand(R,M);
    end
    PUV{i} = Pi;
end

%%ar和beta初始化（arbe表示）
arbe = cell(1,2);
for i = 1:2
    Ni = length(FUV{i});
    arbei = ones(1,Ni)*(1/Ni);
    arbe{i} = arbei;
end


train_tensor = tensor(train_tensor);
normT = norm(train_tensor);
%因子矩阵初始化
N = ndims(train_tensor);
dim_T = size(train_tensor);
Uinit = cell(N,1);
for n = 1:N
    randn('state',n*1000)
    Uinit{n} = sqrt(1/R)*randn(dim_T(n),R);% randomly generate each factor
end
U = Uinit;


maxiters = 10; %AGD算法的迭代次数


du_sum = cell(1,3);
for n=1:3
    du_sum{n} = zeros(size(U{n}));
end

theta = 1;
[curr_log,P,TP] = objv_function(U,FUV,arbe,PUV,option,train_tensor);
objv = curr_log;
eps = 1e-7;
%模型迭代
for iter = 1:maxiters
    disp(iter)
    tic
    %%%%更新U{1},U{2},U{3}
    for k=1:10
        for n = 1:N
            %%%%逻辑张量部分
            DU = mttkrp_opt(train_tensor,TP,P,U,n,c);

%             mT = mttkrp_opt(train_tensor,U,n);
%             mTP = mttkrp_opt(TP,U,n);
%             mP = mttkrp_opt(P,U,n);
%             DU = -c*mT+(c-1)*mTP+mP;
            DU(isnan(DU)) = 0;
            if n<=2
                %%%%%增加特征正则化项
                DU = DU+ar*U{n};
                
                %%%%%增加稀疏子空间学习部分
                for i=1:length(FUV{n})
                    DU = DU+ arbe{n}(i)^yita*(U{n}-FUV{n}{i}*PUV{n}{i}');
                end
                
            else
                
                %%%%%增加特征正则化项
                DU = DU+ar*U{n};
                
            end
            %%%更新
            DU = -DU;
            du_sum{n} = du_sum{n} + DU.^2;   %步骤8中的φ
            
            vec_step_size = theta*ones(size(du_sum{n}))./ sqrt(du_sum{n}+eps);   %迭代步长γ
            vec_step_size(isnan(vec_step_size)) = 0;
            
            U{n} = U{n} + vec_step_size .* DU;    %步骤9
            [last_log,P,TP] = objv_function(U,FUV,arbe,PUV,option,train_tensor);
            objv = [objv,last_log];
        end
    end
    
    for k=1:10
        %%%更新PUV
        for n = 1:2
            for i = length(PUV{n})
                M = size(PUV{n}{i},2);
                UP = U{n}.*(U{n}>=0); UN = -U{n}.*(U{n}<0);
                fenzi = arbe{n}(i)^yita * UP'*FUV{n}{i} ;
                fenmu = arbe{n}(i)^yita * PUV{n}{i}*FUV{n}{i}'*FUV{n}{i}  +...
                    arbe{n}(i)^yita * UN'*FUV{n}{i}+...
                    miu*PUV{n}{i}*ones(M,M);  %ones(M,M)
                PUV{n}{i} = sqrt(fenzi./fenmu).* PUV{n}{i};
                PUV{n}{i}(isnan(PUV{n}{i})) = 0;
            end
        end
        
        %%%%更新arbe
        for n = 1:2
            fan = zeros(1,length(arbe{n}));
            for i = length(arbe{n})
                fan(i) = norm(FUV{n}{i}*PUV{n}{i}'-U{n},'fro')^2;
                fan1(i) = (1/fan(i))^(1/(arbe{n}(i)-1));
            end
            arbe{n} = fan1/sum(fan1);
            if sum(isnan(arbe{n}))>0
                arbe{n} = ones(1,length(arbe{n}))*(1/length(arbe{n}));
            end
        end
    end 
    
 
    
    %计算目标函数的改变量
    delta_log = abs(curr_log-last_log)/abs(last_log);   %改变的相对误差
    
    if abs(delta_log) < 1e-5
        break;
    end
    curr_log = last_log;
    toc
end

%% Clean up final result
% [U{1},U{2}] = Assist_method_Utils.complete_opt(train_tensor,hp_sim,vp_sim,U{1},U{2});
P0 = ktensor(U);
P0 = full(P0);
pre_tensor = exp(P0)./(1+exp(P0));
plot(objv)
end