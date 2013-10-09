#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

export MCMCLikModel, model

### Model types hierarchy to allow restrictions on applicable samplers
abstract Model
abstract MCMCModel <: Model

function ispartition(m::Dict, n::Integer)
	c = zeros(n)
	for v in values(m)
		c[v[1]:(v[1]+prod(v[2])-1)] += 1
	end
	all(c .== 1.)
end

#### misc functions common to all models  ####
hasvectormethod(f::Function) = !isgeneric(f) | length(methods(f, (Vector{Float64},))) == 1
hasgradient{M<:MCMCModel}(m::M) = m.evalg != nothing
hastensor{M<:MCMCModel}(m::M) = m.evalt != nothing
hasdtensor{M<:MCMCModel}(m::M) = m.evaldt != nothing

#### User-facing model creation function  ####

# Currently only a "likelihood" model type makes sense
# Left as is in case other kind of models come up
function model(f::Union(Function, Expr); mtype="likelihood", args...)
	if mtype == "likelihood"
		return MCMCLikelihoodModel(f; args...)
	elseif mtype == "whatever"
	else
	end
end

#### models  #####

include("likmodel.jl")

# Uncommented temporarily the line below in order to run the examples and debug devel
include("bayesglmmodels.jl")  # TODO : not yet adapted


