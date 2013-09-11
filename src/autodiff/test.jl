cd("p:/Documents/julia/MCMC.jl/src/autodiff")
cd("~/devl/MCMC.jl/src/autodiff")
pwd()

@windows_only include("p:/Documents/julia/MCMC.jl/src/autodiff/mymod.jl")
@unix_only include("/home/fredo/devl/MCMC.jl/src/autodiff/mymod.jl")

######



	b = 12
	Abcd.debug(:( a+b ), a=0)
	b = [1, 2, 3]
	Abcd.debug(:( a+b ), a=0)

	Abcd.generateModelFunction(:( x+y ), gradient=true, x=0, debug=true)
	f, a,b,c = Abcd.generateModelFunction(:( x+y ), gradient=true, x=0)
	f, a,b,c = Abcd.generateModelFunction(:( a=x ), gradient=true, x=0)
	f([-13.])

################

# simulate dataset
	srand(1)
	nbeta = 10 # number of predictors, including intercept
	beta0 = randn((nbeta,))

	n = 1000
	X = [ones(n) randn((n, nbeta-1))]
	Y = float64( rand(n) .< ( 1 ./ (1. + exp(X * beta0))) )

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

Abcd.debug(ex, vars=zeros(nbeta))  
Abcd.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)  

ex2 = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = sum(vars) * sum(X)
	prob ~ Normal(1,2)
end
Abcd.debug(ex2, vars=0.0)  # no method Bernoulli(Array{Float64,1},)
