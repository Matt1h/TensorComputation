using DifferentialEquations
using TensorOperations
using TensorComputation
using Lasso
using SparseArrays


function test_model!(dx, x, param, t)
	alpha, beta = param 
	dx[1] = alpha*sin(x[1] - x[2])
    dx[2] = beta*sin(x[1])
end


function CalDerivative(u, dt)
    #  u: Measurement data we wish to approxiamte the derivative. 
    #  It should be of size n x m, where n is the number of measurement, m is the number of states.
    # dt: Time step
    # return du: The approximated derivative. 
    
    # Define the coeficient for different orders of derivative
    p1=1/12;p2=-2/3;p3=0;p4=2/3;p5=-1/12;
    
    du=(p1*u[1:end-4,:]+p2*u[2:end-3,:]+p3*u[3:end-2,:]+p4*u[4:end-1,:]+p5*u[5:end,:])/dt;
        
    return du
end


function optimize_matricized(Theta, dX, lambda)
    p = Theta.order-1
    m = size(X, 1)
    n = size(X, 2)

    Xi = pinv(Theta, p)
    @tensor Xi.cores[Xi.order].a[i, j, k] := Xi.cores[Xi.order].a[i, m, k]*dX[m, j]
    Xi = matricize(Xi)

    Theta = matricize(Theta)'
    for k in range(1, 100)
        smallinds = collect(abs.(Xi) .< lambda)
        Xi[smallinds] .= 0
        for idx in range(1, n)
            biginds = collect(.~smallinds[:, idx])
            Xi[biginds, idx] = Theta[:, biginds]\dX[:, idx]
        end
    end
    return Xi
end


function threshold(A::TT{T}, lamda::Real) where{T}
    A_full = full(A)
    smallinds = collect(abs.(A_full) .< lamda)
    smallinds_TT = TT(smallinds)
    for i in range(1, A.order)
        A.cores[i].a[findall(smallinds_TT.cores[i].a .== 0)] .= T(0)
    end
    return A
end


function optimize_TT(Theta, dX, lambda)
    p = Theta.order-1
    m = size(dX, 1)
    n = size(dX, 2)

    Xi = pinv(Theta, p)
    @tensor Xi.cores[Xi.order].a[i, j, k] := Xi.cores[Xi.order].a[i, m, k]*dX[m, j]
    Xi.RowDims[Xi.order] = size(Xi.cores[Xi.order], 2)
    # Xi_M = matricize(Xi)
    Xi_full = full(Xi)
    # smallinds = TT(collect(abs.(Xi_full) .< lambda))

    # for k in range(1, 100)
    #     Xi_full = full(Xi)
    #     smallinds = TT(collect(abs.(Xi_full) .< lambda))
    #     # Xi[smallinds] .= 0

    #     for idx in range(1, n)
    #         biginds = collect(.~smallinds[:, idx])
    #         Xi[biginds, idx] = Theta[:, biginds]\dX[:, idx]
    #     end

    # end
end


# test model parameters
tmax = 20
dt = 0.01
x0 = [2.0; 1.8]
tspan = (0.0, tmax)
alpha = 17.0
beta = 5.0
param = (alpha, beta)

# generate data
prob = ODEProblem(test_model!, x0, tspan, param)
sol = solve(prob, saveat=dt)
X = Array(solve(prob,saveat=dt))'
dX = CalDerivative(X, dt)
X = X[3:end-2,:]

# set Tensor Train
psi = [sin, cos]
Theta = TT(X', psi, "function_major")

# calculate Xi
p = Theta.order-1
m = size(X, 1)
Xi = pinv(Theta, p)
@tensor Xi.cores[Xi.order].a[i, j, k] := Xi.cores[Xi.order].a[i, m, k]*dX[m, j]  # TODO: this should be a function that changes RowDims too
Xi.RowDims[Xi.order] = size(Xi.cores[Xi.order].a, 2)

# lamda1 = 0.1
# lamda2 = 0.000001
# A = Xi
# A_full = full(A)
# smallinds1 = collect(abs.(A_full) .< lamda1)
# smallinds2 = collect(abs.(A_full) .< lamda2)
# smallinds_TT1 = TT(smallinds1)
# smallinds_TT2 = TT(smallinds2)
# idx1 = findall(smallinds_TT1.cores[1] .== 1)
# idx2 = findall(smallinds_TT2.cores[1] .== 1)
# for i in range(1, A.order)
#     A.cores[i][findall(smallinds_TT1.cores[i] .== 1)] .= 0.0
# end
# matricize(A)

Xi_M = matricize(Xi)
Xi = matricize(threshold(Xi, 0.1))


# Xi_M = optimize_TT(Theta, dX, 0.01)

# # reconstruct derivatives
# Xi_M = matricize(Xi)
# Theta = matricize(Theta)'
# Y = Theta*Xi

# # fit Data in Matrixformat with Lasso
# f = fit(LassoPath, Matrix(matricize(Theta)'), dX[:, 2], standardize = false, intercept = false, λ=[0.001; 0.0001; 0.00001; 0.000001])
# f = fit(LassoPath, Matrix(matricize(Theta)'), dX[:, 1], λ=[0.001; 0.0001; 0.00001; 0.000001])
# f.coefs