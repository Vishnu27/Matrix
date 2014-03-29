function [fd,fval,mv] = fderreal(rB,iB,W,R,z,gamma,q,tol)
% Compute the derivative of the 2nd singular value using the Lanczos
% algorithm

[m,n] = size(R);

if gamma == 0
    error('??? gamma should be strictly larger than 0');
end

if ~isempty(rB),
   % Do explicit singular value computation
   mv = 0;
   C = [rB -1/gamma*iB; gamma*iB rB];
   [U,singval,V] = svd(C);
   Cdot = [zeros(m,m) (1/gamma^2)*iB; iB zeros(m,m)];
   fd = real(U(:,2)'*Cdot*V(:,2));
   fval = singval(2,2);

else
   % Avoiding explicit inversion of B
   [VV,T,res,mv] = lanczsvdreal(W,R,z,gamma,2*m,q,tol);

   % Approximate left and right singular vectors
   [Q,Theta] = eig(T);

   sigma2 = sqrt(Theta(end-1,end-1));
   Y = VV*Q;
   v2 = Y(:,end-1);
   x = W'*(v2(1:m) + i*(1/gamma)*v2(m+1:2*m));
   x = (R-z*eye(size(R)))\x;
   x = W*x;
   u2(1:n) = real(x);
   u2(n+1:2*n) = gamma * imag(x);
   u2 = u2';
   u2 = u2/sigma2;

   %Cdot*V(:,2): 1st part
   t = W'*(zeros(n,1) -i*(1/gamma^2)*v2(n+1:end) );
   t = (R-z*eye(size(R)))\t;
   t = W*t;
   t1 = real(t);
   %Cdot*V(:,2): 2nd part
   t = W'*(zeros(n,1) -i*v2(1:n) );
   t = (R-z*eye(size(R)))\t;
   t = W*t;
   t2 = real(t);
   
   % Approximation to the derivative of sigma2
   fd = real(u2'*[t1;t2]);
   fval = sigma2;
end