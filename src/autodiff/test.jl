
@windows_only include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/Autodiff.jl")
@unix_only include("/home/fredo/devl/MCMC.jl.fredo/src/autodiff/Autodiff.jl")


ex = quote
	a = b+56
	a = a + exp(X)
	res = sum(a)
end

X = [1. 2 ; 0  4]
res = Autodiff.diff(ex, :res, b=12)

using Distributions
a = [ Normal(1,2), Gamma(1,1)]
a = [ i<5 for i in 1:10]

isa(a,Array)
isa(a,AbstractArray)
isa(a,Array)
isa(a,Array{Bool})
isa(a,AbstractArray{Bool})
isa(a,Array{Distribution})
isa(a,AbstractArray{Distribution})

ty{T<:Distribution}(a::Array{T}) = "yes"
ty{T<:Distribution}(a::AbstractArray{T}) = "yesyes"
ty(a::Array{Distribution}) = "ad"
ty(a::AbstractArray{Distribution}) = "aad"
ty(a) = "no"

ty(a)

dump(:(ty{T<:Distribution}(a::Array{T}) = "yes"),10)




