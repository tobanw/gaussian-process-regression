using LinearAlgebra, Distributions, Plots
include("GaussianProcesses.jl")

# distance function (squared)
# for euclidean, just use norm(v, p)
d_euclid(a::Array, b::Array) = norm(a-b, 2)

# example function
f_target(x) = sqrt(x) * sin(x/10)
noise_distro = Normal(0.8)

x_train = collect(range(0, 100, length = 20));
#= x_train = [collect(range(0, 49, length = 50))..., collect(range(50, 100, length = 10))...]; =#
y_train = f_target.(x_train) .+ rand.(noise_distro);
plot(x_train, y_train)

# plotting grid
samplex = collect(range(0, 100, length = 10));

# instantiate a GP
myGP = GaussianProcesses.GP_Regressor(x_train, y_train, samplex; σ=0.8, h=20.0, d=d_euclid);

# draw one sample function from prior
plot(myGP.x_grid, rand(myGP.prior))

# check posterior mean
plot(myGP.x_grid, mean(myGP.posterior))

# check posterior standard deviation
plot(myGP.x_grid, sqrt.(diag(cov(myGP.posterior))))

# plot equivalent kernel weights of train data
plot(myGP.x_train, myGP.equiv_weights[10,:])


# multidimensional example

using DataFrames

dt = DataFrame(
	x1=[-5,-5,-5,0,0,0,5,5,5],
	x2=[-5,0,5,-5,0,5,-5,0,5]
);

f(x1,x2) = -x1^2 - x2^2

dt[!, :y] .= f.(dt.x1, dt.x2);
dt
