function DU = mttkrp_opt(X1,X2,X3,U,n,c)
%X1表示Y,trian_tensor
%X2表示P*Y
%X3表示P
%MTTKRP Matricized tensor times Khatri-Rao product for tensor.
%
%   V = MTTKRP(X,U,N) efficiently calculates the matrix product of the
%   n-mode matricization of X with the Khatri-Rao product of all
%   entries in U, a cell array of matrices, except the Nth.  How to
%   most efficiently do this computation depends on the type of tensor
%   involved.
%
%   V = MTTKRP(X,K,N) instead uses the Khatri-Rao product formed by the
%   matrices and lambda vector stored in the ktensor K. As with the cell
%   array, it ignores the Nth factor matrix. The lambda vector is absorbed
%   into one of the factor matrices.
%
%   NOTE: Updated to use BSXFUN per work of Phan Anh Huy. See Anh Huy Phan,
%   Petr Tichavsk�, Andrzej Cichocki, On Fast Computation of Gradients for
%   CANDECOMP/PARAFAC Algorithms, arXiv:1204.1586, 2012.
%
%   Examples
%   mttkrp(tensor(rand(3,3,3)), {rand(3,3), rand(3,3), rand(3,3)}, 2)
%   mttkrp(tensor(rand(2,4,5)), {rand(2,6), rand(4,6), rand(5,6)}, 3)
%
%   See also TENSOR, TENMAT, KHATRIRAO
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

% Multiple versions supported...


N = ndims(X1);
R = size(U{n},2);
%% Computation

szl = prod(size(X1,1:n-1)); %#ok<*PSIZE>
szr = prod(size(X1,n+1:N));
szn = size(X1,n);

if n == 1
    Ur = khatrirao(U{2:N},'r');
    XX = -c*X1+(c-1)*X2+X3;
    YY = reshape(XX.data,szn,szr);
    DU = YY * Ur;
%     
%     
%     
%     Y1 = reshape(X1.data,szn,szr);
%     V1 =  Y1 * Ur;
%     
%     Y2 = reshape(X2.data,szn,szr);
%     V2 =  Y2 * Ur;   
%     
%     Y3 = reshape(X3.data,szn,szr);
%     V3 =  Y3 * Ur;      
elseif n == N
    Ul = khatrirao(U{1:N-1},'r');
    XX = -c*X1+(c-1)*X2+X3;
    YY = reshape(XX.data,szl,szn);
    DU = YY' * Ul;
    
%     Y1 = reshape(X1.data,szl,szn);
%     V1 = Y1' * Ul;
%     
%     Y2 = reshape(X2.data,szl,szn);
%     V2 = Y2' * Ul;
%     
%     Y3 = reshape(X3.data,szl,szn);
%     V3 = Y3' * Ul;
else
    Ul = khatrirao(U{n+1:N},'r');
    Ur = reshape(khatrirao(U{1:n-1},'r'), szl, 1, R);
    XX = -c*X1+(c-1)*X2+X3;
    Y1 = reshape(XX.data,[],szr);
    Y1 = Y1 * Ul;
    Y1 = reshape(Y1,szl,szn,R);    
    DU = zeros(szn,R);
    for r =1:R
        DU(:,r) = Y1(:,:,r)'*Ur(:,:,r);
    end
end

end

