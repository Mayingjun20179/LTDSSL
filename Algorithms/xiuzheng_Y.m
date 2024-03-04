%%%%关于连接矩阵处理(对于整个矩阵进行处理)
function Y = xiuzheng_Y(Y0,SHV)  %ar为衰减系数

simH = SHV{1};
simV = SHV{2};
Yh0 = tenmat(Y0,1);  Yh = double(Yh0);
Yv0 = tenmat(Y0,2);  Yv = double(Yv0);
K = 50;
ar = 1; 

flagh = sum(Yh,2);    ind_h0 = find(flagh==0);    ind_h1 = find(flagh>0);   
flagv = sum(Yv,2);   ind_v0 = find(flagv==0);    ind_v1 = find(flagv>0);     



%%
Nh = size(Yh,1);
simH(1:(Nh+1):end) = 0;      %对角线变为0

Nv = size(Yv,1);
simV(1:(Nv+1):end) = 0; 

simH(ind_h0,ind_h0) = 0;       %单独疾病和代谢的相似度为0
simV(ind_v0,ind_v0) = 0;

[~,indh] = sort(simH,2,'descend');

ar = ar.^(0:K-1);
for i=1:length(ind_h0)    
    near_inh = indh(ind_h0(i),1:K);
    Yh(ind_h0(i),:) = (ar.*simH(ind_h0(i),near_inh))*Yh(near_inh,:)/sum((ar.*simH(ind_h0(i),near_inh)));
end

Yh =  tenmat(Yh,Yh0.rdims,Yh0.cdims,Yh0.tsize);
Yh = tensor(Yh);

[~,indv] = sort(simV,2,'descend');
ar = ar.^(0:K-1);
for j=1:length(ind_v0)   
    near_inv = indv(ind_v0(j),1:K);
    Yv(ind_v0(j),:) = (ar.*simV(ind_v0(j),near_inv))*Yv(near_inv,:)/sum((ar.*simV(ind_v0(j),near_inv)));
end 
Yv =  tenmat(Yv,Yv0.rdims,Yv0.cdims,Yv0.tsize);
Yv = tensor(Yv);

Y = (Yh+Yv)/2;

end