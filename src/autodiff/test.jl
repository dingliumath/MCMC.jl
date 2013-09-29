
@windows_only include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/Autodiff.jl")
@unix_only include("/home/fredo/devl/MCMC.jl.fredo/src/autodiff/Autodiff.jl")


m = Autodiff.ParsingStruct()
m.source = :( a = 45 ; b += 45; copy!(b,a); c = X')
Autodiff.unfold!(m)
m.exprs



ex = quote
	a = b+56
	a = a + exp(X)
	res = sum(a)
end

X = [1. 2 ; 0  4]
res = Autodiff.diff(ex, :res, b=12)

Autodiff.diff(:( copy!(a,b) ), :a, b=1)

######################

srand(1)
n = 1000
nbeta = 10 # number of covariates, including intercept

X = [ones(n) randn((n, nbeta-1))]  # covariates

beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(-X * beta0)))

# define model
ex = quote
    acc = sum( logpdf(Normal(0,1), vars) )
    prob = 1 / (1. + exp(-X * vars)) 
    acc = acc + sum(logpdf(Bernoulli(prob), Y))
end

head, body, rsym = Autodiff.diff(ex, :acc, vars=zeros(nbeta))

using Distributions
using Base.LinAlg.BLAS

Distributions.Normal(1,1)

include("myf.jl")

myf(ones(nbeta))
myf10000() = (for i in 1:1000 ; myf(ones(nbeta)) ; end )

using Profile

myf10000()
sprofile_clear()
@sprofile myf10000()
sprofile_tree()






