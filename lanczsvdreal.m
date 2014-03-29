function [V,T,res,k] = lanczsvdreal(U,R,z,gamma,m,v0,tol)
%
% [V,T,res] = lanczsvdreal(U,R,z,gamma,m,v0,tol)
%
% Approximate the second largest singular triplets of 
%  
%        | real(B)           -1/gamma * imag(B) |
%    C = |                                      |,
%        | gamma* imag(B)    real(B)            |
%  
% where B = inv(A-z*I), without explicitly forming B. The matrix C
% shows up in the computation of REAL pseudospectra.
%
% Input arguments: 
% [U,R] : schur decomposition of A
% z     : complex shift
% gamma : the scalar $\gamma$
% m     : maximum number of Lanczos steps
% v0    : starting vector
% tol   : tolerance of the residual
%
% Output arguments:
% V   : the Lanczos basis
% T   : the tridiagonal matrix
% res : the residual history


% Size of A
n = size(U,1);

% normalize v0
V(:,1) = v0/norm(v0);
v_prev = zeros(2*n,1);
w = zeros(2*n,1);
beta  = 0;
for k=1:m
    % Multiplication by C
    x = U'*(V(1:n,k) + i*(1/gamma)*V(n+1:2*n,k));
    x = (R-z*eye(size(R)))\x;
    x = U*x;
    w(1:n) = real(x);
    w(n+1:2*n) = gamma * imag(x);    
    % Multiplication by C'    
    x = U'*(w(1:n) + i*gamma*w(n+1:2*n));
    x = (R'-z'*eye(size(R)))\x;
    x = U*x;
    w(1:n) = real(x);
    w(n+1:2*n) = imag(x)/gamma;
    
    w = w - beta * v_prev;
    alpha = w'*V(:,k);
    w = w - alpha * V(:,k);

    % Full reorthogonalization
    s = V'*w;
    for l=1:k
       w = w - s(l)*V(:,l);
    end

    T(k,k+1) = norm(w);
    T(k+1,k) = T(k,k+1);
    V(:,k+1) = w / T(k,k+1);

    v_prev = V(:,k);
    beta = T(k+1,k);
    T(k,k) = alpha;
    
    % Check for convergence
    if (k>=2)
        [S,theta] = eig(T(1:k,1:k));
        [list,index] = sort(diag(theta));
        res(k) = abs(beta*S(end,index(end-1)));
        if (res(k) <=tol)
            str = sprintf('Lanczos has converged in %d steps.',k);
            %disp(str)
            break;
        end
    end
end
