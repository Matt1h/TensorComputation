using TensorComputation

psi = [sin, cos]
X = [3 4 1; 
    1 1.2 2; 
    1 2 3; 
    0 9 2]
X = X'

M = TT(X, psi, "function_major")
P = pinv(M, M.order-1)
MM = matricize(M)
