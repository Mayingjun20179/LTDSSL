function [L,SN] = Calculate_Laplace(S,knn)
n = size(S,1);
SN = zeros(n,n);
%
for i=1:n
    ll = S(i,:);
    [~,index_i] = sort(ll,'descend');
    k_ii = index_i(1:knn);
    SN(i,k_ii) = S(i,k_ii);  %
end
SN = (SN+SN')/2;

%
D = sum(SN,2);
D2 = diag(D.^(-1/2));
D2(isnan(D2)) = 0;
L = D2*(D-SN)*D2';



end