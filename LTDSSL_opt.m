%%%%
function pre_tensor = LTDSSL_opt(train_tensor, FUV,option)
R = option(1);  %
ar = option(2);  %
c = option(3);
yita = option(4);
miu = option(5);

%PUV
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

%%ar,beta
arbe = cell(1,2);
for i = 1:2
    Ni = length(FUV{i});
    arbei = ones(1,Ni)*(1/Ni);
    arbe{i} = arbei;
end


train_tensor = tensor(train_tensor);
normT = norm(train_tensor);
%
N = ndims(train_tensor);
dim_T = size(train_tensor);
Uinit = cell(N,1);
for n = 1:N
    randn('state',n*1000)
    Uinit{n} = sqrt(1/R)*randn(dim_T(n),R);% randomly generate each factor
end
U = Uinit;


maxiters = 10; %


du_sum = cell(1,3);
for n=1:3
    du_sum{n} = zeros(size(U{n}));
end

theta = 1;
[curr_log,P,TP] = objv_function(U,FUV,arbe,PUV,option,train_tensor);
objv = curr_log;
eps = 1e-7;
%
for iter = 1:maxiters
    disp(iter)
    %%%%U{1},U{2},U{3}
    for k=1:10
        for n = 1:N
            %%%%
            DU = mttkrp_opt(train_tensor,TP,P,U,n,c);
            DU(isnan(DU)) = 0;
            if n<=2
                %%%%%
                DU = DU+ar*U{n};
                
                %%%%%
                for i=1:length(FUV{n})
                    DU = DU+ arbe{n}(i)^yita*(U{n}-FUV{n}{i}*PUV{n}{i}');
                end
            else                
                %%%%%
                DU = DU+ar*U{n};
                
            end
            %%%
            DU = -DU;
            du_sum{n} = du_sum{n} + DU.^2;   %
            
            vec_step_size = theta*ones(size(du_sum{n}))./ sqrt(du_sum{n}+eps);   %迭代步长γ
            vec_step_size(isnan(vec_step_size)) = 0;
            
            U{n} = U{n} + vec_step_size .* DU;    %
            [last_log,P,TP] = objv_function(U,FUV,arbe,PUV,option,train_tensor);
            objv = [objv,last_log];
        end

    end
    
    for k=1:10
        %%%PUV
        for n = 1:2
            for i = 1:length(PUV{n})
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
        
        %%%%arbe
        for n = 1:2
            fan1 = zeros(1,length(arbe{n}));
            for i = 1:length(arbe{n})
                fan = norm(FUV{n}{i}*PUV{n}{i}'-U{n},'fro')^2;
                fan1(i) = fan^(1/(1-yita));
            end
            arbe{n} = fan1/sum(fan1);
            if sum(isnan(arbe{n}))>0
                arbe{n} = ones(1,length(arbe{n}))*(1/length(arbe{n}));
            end
        end
    end 
    
    delta_log = abs(curr_log-last_log)/abs(last_log);   
    if abs(delta_log) < 1e-5
        break;
    end
    curr_log = last_log;
end

%% Clean up final result
P0 = ktensor(U);
P0 = full(P0);
pre_tensor = exp(P0)./(1+exp(P0));
plot(objv)
end
