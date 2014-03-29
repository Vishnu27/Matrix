input C1
inpur r
b(1)=norm(r,2);

for k= 2:4
    
    q(k,:)=r/b(k-1);
    r=C1*q;
    r=r-q(k-1,:)*b(k-1);
    a(k)=dot(q(k,:),r);
    for l=1:k-1
        r=r-(dot(r,q(l,:))*q(l,:));
    end
    b(k)=norm(r,2);
end

        