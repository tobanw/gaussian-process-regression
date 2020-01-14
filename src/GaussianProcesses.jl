module GaussianProcesses

export k, K, GP_Regressor

using LinearAlgebra, Distributions

# exponential quadratic kernel
k(a::Array, b::Array; h::Real, d::Function) = exp(-d(a,b)^2 / (2 * h^2))
k(a::Real, b::Real; h::Real, d::Function) = k([a], [b]; h=h, d=d)

# kernel matrix
K(A::Vector, B::Vector; h::Real, d::Function) = [k(a, b; h = h, d = d) for a in A, b in B]

# prior distribution: mean 0, covariance K
mutable struct GP_Regressor
	"MVN distribution of prior"
	prior::Distributions.MvNormal
	"MVN distribution of posterior"
	posterior::Distributions.MvNormal
	"equivalent smoothing kernel weights (rows of matrix)"
	equiv_weights::Array
	"input points for sampling and plotting"
	x_grid::Vector
	"sample observations"
	x_train::Vector
	"sample outcomes"
	y_train::Vector
	"standard deviation of iid Gaussian noise in observations"
	σ::Real
	"characteristic length scale"
	h::Real
	"distance function"
	d::Function

	# inner constructor: call this function to instantiate
	function GP_Regressor(X_train::Vector, y_train::Vector, X::Vector; σ::Real, h::Real, d::Function)

		# prior
		Kpred = K(X, X; h=h, d=d)  # covariance
		prior = MvNormal(Kpred)

		# posterior terms
		Kinv = inv(K(X_train, X_train; h=h, d=d) + σ^2 * I)
		Kcross = K(X, X_train; h=h, d=d)
		Kequiv = Kcross * Kinv  # equivalent smoothing kernel weights (each row)

		mean_posterior = Kequiv * y_train
		cov_posterior = Kpred .- Kcross * Kinv * Kcross'

		posterior = MvNormal(mean_posterior, Matrix(Hermitian(cov_posterior)))

		new(prior, posterior, Kequiv, X, X_train, y_train, σ, h, d)
	end
end

end  # module
