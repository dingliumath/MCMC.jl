#################################################################
#
#    Definition of Likelihood-type model
#
#   (Basic MCMC model type based on evaluating the log-target)
#
#   Examples of other possible models: MCMCHierarchicalModel, 
#      MCMCGPModel, MCMCKernelModel
#
#################################################################

export MCMCLikModel

println("Loading MCMCLikModel model")

typealias ROrVector     Union(Real, Vector{Float64})
typealias FOrNothing    Union(Nothing, Function)

# The likelihood Model type
type MCMCLikelihoodModel <: MCMCModel
	eval::Function              # log-likelihood evaluation function
	evalg::Union(Nothing, Function) 			# gradient vector evaluation function
	evalt::Union(Nothing, Function) 			# tensor evaluation function
	evaldt::Union(Nothing, Function) 			# tensor derivative evaluation function
  	evalallg::Union(Nothing, Function) 		# 2-tuple (log-lik, gradient vector) evaluation function
  	evalallt::Union(Nothing, Function) 		# 3-tuple (log-lik, gradient vector, tensor) evaluation function
  	evalalldt::Union(Nothing, Function) 		# 4-tuple (log-lik, gradient vector, tensor, tensor derivative) evaluation function
	pmap::Dict                  # map to/from parameter vector from/to user-friendly variables
	size::Integer               # parameter vector size
	init::Vector{Float64}       # parameter vector initial values
	scale::Vector{Float64}      # scaling hint on parameters

	MCMCLikelihoodModel(f::Function, 
						g::Union(Nothing, Function), ag::Union(Nothing, Function),
						t::Union(Nothing, Function), at::Union(Nothing, Function),
						dt::Union(Nothing, Function), adt::Union(Nothing, Function),
						i::Vector{Float64}, 
						sc::Vector{Float64}, 
						pmap::Dict) = begin

		s = size(i, 1)

		@assert ispartition(pmap, s) "param map is not a partition of parameter vector"
		@assert size(sc,1) == s "scale parameter size ($(size(sc,1))) different from initial values ($s)"

		# check that all functions can be called with a vector of Float64 as argument
		for ff in [f, g, ag, t, at, dt, adt]
			@assert ff==nothing || hasvectormethod(f) 
					"one of the supplied functions cannot be called with Vector{Float64}"
			#TODO : make error message print which function is problematic
		end

		# check that initial values are in the support of likelihood function
		@assert isfinite(f(i)) "Initial values out of model support, try other values"

		new(f, g, t, dt, ag, at, adt, pmap, s, i, sc)
	end
end

typealias MCMCLikModel MCMCLikelihoodModel

# Model creation using expression parsing and autodiff
function MCMCLikelihoodModel(	m::Expr; 
								gradient::Bool=false,
								init=nothing,
								pmap=nothing,
								scale::Union(Real, Vector{Float64}) = 1.0,
								args...)
	# when using expressions, initial values are passed in keyword args
	#  with one arg by parameter, therefore there is not need for an init arg
	init == nothing || error("'init' kwargs not allowed for model as expression\n")

	# same thing with 'pmap'
	pmap == nothing || error("'pmap' kwargs not allowed for model as expression\n")

	# generate log-lik function
	f, s, p, i = generateModelFunction(m; gradient=false, args...) # loglik only function

	# generate gradient function if requested
	if gradient
		g, s, p, i = generateModelFunction(m; gradient=true, args...) # loglik and gradient function
	else
		g = nothing
	end

	MCMCLikelihoodModel(f, allgrad=g, init=i, pmap=p, scale=scale)
end


# Model creation : with user supplied functions
function MCMCLikelihoodModel(	lik::Function;
								grad::Union(Nothing, Function) = nothing, 
								tensor::Union(Nothing, Function) = nothing,
								dtensor::Union(Nothing, Function) = nothing,
								allgrad::Union(Nothing, Function) = nothing, 
								alltensor::Union(Nothing, Function) = nothing,
								alldtensor::Union(Nothing, Function) = nothing,
								init::Union(Real, Vector{Float64}) = [1.0], 
								scale::Union(Real, Vector{Float64}) = 1.0,
								pmap::Union(Nothing, Dict) = nothing) 

	# convert init to vector if needed
	init = isa(init, Real) ? [init] : init

	# expand scale to parameter vector size if needed
	scale = isa(scale, Real) ? scale * ones(length(init)) : scale

	# all parameters named "pars" by default
	if pmap == nothing ; pmap = Dict([:pars], [(1, size(init))]) ; end 

	# now build missing functions, if any
	fmat = Array(Union(Nothing, Function), 3, 2)
	for (i, f1, f2) in [(1, grad, allgrad), 
						(2, tensor, alltensor), 
						(3, dtensor, alldtensor)]
		fmat[i,:] = [f1 f2]
		if f1==nothing && f2!=nothing # only the tuple version is supplied
			fmat[i,1] = (v) -> f2(v)[end] 
		elseif f1!=nothing && f2==nothing # only the single version is supplied
			if i == 1
				fmat[i,2] = (v) -> (lik(v), f1(v))
			else
				@assert isa(fmat[i-1,2], Function) "missing function !"
				fmat[i,2] = (v) -> tuple(fmat[i-1,2](v)..., f1(v))
			end
		end
	end

	MCMCLikelihoodModel(lik, 
						fmat[1,1], fmat[1,2],
						fmat[2,1], fmat[2,2],
						fmat[3,1], fmat[3,2],
						init, scale, pmap)
end
