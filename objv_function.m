function [obj,P,TP] = objv_function(U,FUV,arbe,PUV,option,train_tensor)

R = option(1);  %
ar = option(2);  %
c = option(3);
yita = option(4);
miu = option(5); 


P0 = ktensor(U);
P0 = full(P0);

P = exp(P0)./(1+exp(P0));
TP = P.*train_tensor;

objv1 = (1+(c-1)*train_tensor).*log(1+exp(double(P0)))-...
    c*P0.*train_tensor;
obj = sum(objv1(:));

%
for n = 1:3
    obj = obj+ar/2*(norm(U{n},'fro'));
end

%
for n = 1:2
    for i = 1:length(FUV{n})
        obj = obj+(arbe{n}(i)^yita)/2 * norm(FUV{n}{i}*PUV{n}{i}'-U{n},'fro').^2;
        obj = obj+miu/2*norm(sum(PUV{n}{i},2),'fro')^2;
    end
end


end

