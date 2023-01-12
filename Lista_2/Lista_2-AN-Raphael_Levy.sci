function [A, b] = Ab(n)
    
a1 = zeros(n,2);
v1 = zeros(n,1);
for i=1:n
    a1(i,:) = [i,i];
    v1(i) = 3;
end

a2 = zeros(n-1,2);
v2 = zeros(n-1,1);
for i=2:n
    a2(i-1,:) = [i,i-1];
    v2(i-1) = -1;
end

a3 = zeros(n-1,2);
v3 = zeros(n-1,1);
for i=2:n
    a3(i-1,:) = [i-1,i];
    v3(i-1) = -1;
end

a4 = zeros(n,2);
v4 = zeros(n,1);
for i=1:n
    a4(i,:) = [i,n+1-i];
    v4(i) = 0.5;
end

a4 = cat(1, a4(1:(n/2)-1,:), a4((n/2)+2:n,:));
v4 = cat(1, v4(1:(n/2)-1), v4((n/2)+2:n));

a = cat(1,a1,a2,a3,a4);
v = cat(1,v1,v2,v3,v4);

A = sparse(a,v);    

/////////////////////////////////

b = zeros(n,1);
    b(1) = 2.5;
    b(n) = 2.5;
    b(n/2) = 1;
    b((n/2)+1) = 1;
    for i=2:n/2-1
        b(i) = 1.5;
    end
    for i=n/2+2:n-1
        b(i) = 1.5;
    end

endfunction

function [xk,k]=Jacobi_Method_AN(n,tol)
    
[A, b] = Ab(n);    

k = 0;
xk = zeros(n,1);
xstar = ones(n,1);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Método de Jacobi ////////////////////////////////

D=diag(diag(A)) //Diagonal de A
LPU = A-D //L plus U
invD=diag(1./diag(D))  //Inversa da matriz D
MJ=(-1)*invD*LPU //Matriz de Jacobi
cJ=invD*b //Vetor de Jacobi 


N = %inf 
while N>tol
    xk1=MJ*xk+cJ
    N=norm(xstar-xk1, %inf ) //Norma da diferença entre as aproximações
    xk=xk1
    k=k+1 
end


endfunction

//////////////////////////////////////////////////////////////////////////////////////////////////////

function [x] = Resolve_Lx(L, b)

[t] = size(L, 1);
x = zeros(t, 1);

// Calcula x, sendo Lx=b

x(1) = b(1)/full(L(1, 1));
for i = 2:t,
    x(i) = (b(i) - full(L(i,1:i - 1)) * x(1:i - 1))/full(L(i, i));
end

endfunction

function [xk,k]=Gauss_Seidel_Method_AN(n,tol)

[A, b] = Ab(n);    

xk = zeros(n,1);
xstar = ones(n,1);

////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Método de Gauss-Seidel ////////////////////////////////

D=diag(diag(A)) //Diagonal de A
L=tril(A, -1)
U=triu(A, 1)


N = %inf 
k = 0;
while N>tol
    xk=Resolve_Lx(L+D,b-U*xk)
    N=norm(xstar-xk,%inf) //norma da diferença entre as aproximações
    k=k+1  
end


endfunction

//////////////////////////////////////////////////////////////////////////////////////////////////////

function [xk,N]=Gradiente_Conjugado(n,k)
    
[A, b] = Ab(n);        
    
xk = zeros(n,1);
xstar = ones(n,1);   

//////////////////////////////////////////////////////////////////////////////////
///////////////////////// Método do Gradiente Conjugado //////////////////////////

r = b - A*xk;
p = r;
rsold = r' * r;

for i = 1:k,
    Ap = A * p;
    alpha = rsold / (p' * Ap);
    xk = xk + alpha * p;
    r = r - alpha * Ap;
    rsnew = r' * r;
    if sqrt(rsnew) < 1e-10,
        break;
    end
    p = r + (rsnew / rsold) * p;
    rsold = rsnew;
end

N = norm(xstar - xk, %inf);

endfunction

