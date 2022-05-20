using Distributions, DifferentialEquations, Plots
using TensorOperations
using TensorComputation


function kuramoto!(dθ, θ, param, t)
	N = length(θ)
	K, ω = param 
    for i in 1:N
		dθ[i] = ω[i] + K*mean(sin(θ[j] - θ[i]) for j in 1:N)
	end
end;


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


# kuramoto parameters
n = 100
tmax = 1000
dt = 1
θ₀ = 2*π * collect(range(0,stop=1,length=n)) .- π
tspan = (0.0, tmax)
K = 2
ω = collect(range(-5,stop=5,length=n))
param = (K, ω)

# generate kuramoto data
prob = ODEProblem(kuramoto!, θ₀, tspan, param)
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
@tensor Xi.core[Xi.order][i, j, k] := Xi.core[Xi.order][i, m, k]*dX[m, j]

# reconstruct derivatives
Xi = matricize(Xi)
Theta = matricize(Theta)'
Y = Theta*Xi