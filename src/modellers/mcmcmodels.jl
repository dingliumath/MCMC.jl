#################################################################
#
#    Definition of MCMCModel types 
#
#################################################################

export model

######### parameters map info  ############
# These types are used to map scalars in the
#   parameter vector to user facing variables
#
# immutable PDims
# 	pos::Integer   # starting position of parameter in the parameter vector
# 	dims::Tuple    # dimensions of user facing parameter, can be a scalar, vector or matrix
# end
# typealias PMap Dict{Symbol, PDims}

function ispartition(m::PMap, n::Int)
	c = zeros(n)
	for v in values(m)
		c[v.pos:(v.pos+prod(v.dims)-1)] += 1
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
