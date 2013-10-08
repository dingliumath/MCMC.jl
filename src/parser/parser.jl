##########################################################################
#
#    Model Expression parsing
#      - transforms MCMC specific idioms (~) into regular Julia syntax
#      - calls Autodiff module for gradient code generation
#      - creates function
#
##########################################################################

using Distributions
using Base.LinAlg.BLAS

include("autodiff/Autodiff.jl")
using .Autodiff

# Distributions extensions, TODO : ask for inclusion in Distributions package
include("vector_dists.jl")

#### Log-likelihood accumulator type  ####
# this makes the model function easier to generate compared to a Float64
#   - embeds the error throwing when log-likelihood reaches -Inf
#   - calculates the sum when logpdf() returns an Array
type OutOfSupportError <: Exception ; end

immutable LLAcc
	val::Float64
	function LLAcc(x::Real)
		isfinite(x) || throw(OutOfSupportError())
		new(x)
	end
end
+(ll::LLAcc, x::Real)           = LLAcc(ll.val + x)
+(ll::LLAcc, x::Array{Float64}) = LLAcc(ll.val + sum(x))
declareType(LLAcc, :LLAcc) # declares new type to Autodiff
@deriv_rule getfield(x::LLAcc, f      )      x     dx1 = ds

#### add new derivation rules to Autodiff
include("MCMC_deriv_rules.jl")

# naming conventions
const ACC_SYM = :__acc       # name of accumulator variable
const PARAM_SYM = :__beta    # name of parameter vector


#######################################################################
#   generates the log-likelihood function
#######################################################################
# - 'init' contains the dictionary of model params and their initial value
# - If 'debug' is set to true, the function returns only the function expression
#  that would have been created
function generateModelFunction(model::Expr; gradient=false, debug=false, init...)

	model.head != :block && (model = Expr(:block, model))  # enclose in block if needed
	length(model.args)==0 && error("model should have at least 1 statement")

	model = translate(model)
	model = Expr(:block, [ :($ACC_SYM = LLAcc(0.)), 
		                   model.args, 
		                   :( $ACC_SYM = $(Expr(:., ACC_SYM, Expr(:quote, :val)) ) )]... )

	resetvar()  # reset temporary variable numbering (for legibility, not strictly necessary)
	head, body, outsym = Sandbox.diff(model, ACC_SYM; init...)
	println(head)
	println(body)

	## build function expression
	if gradient  # case with gradient
		body = [ betaAssign(m)...,              # assigments beta vector -> model parameter vars
		         body]

		# return statement (note : gradient vec should match beta variable mapping)
		# dexp = { :( vec([$(dsym(p.sym))]) ) for p in m.pars}
		gsym = dprefix(PARAM_SYM)
		push!(body, :( local $gsym = similar($PARAM_SYM)))
		for p in keys(m.pars)
			v = m.pars[p]
			dsymp = dprefix(p)

			if length(v.dims) == 0  # scalar
				push!(body, :( $gsym[ $(v.pos) ] = $dsymp ) )
			elseif length(v.dims) == 1  # vector
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(body, :( $gsym[ $(Expr(:quote,r)) ] = $dsymp ) )
			else # matrix case  (needs a reshape)
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(body, :( $gsym[ $(Expr(:quote,r)) ] = vec($dsymp) ))
			end
		end
		push!(body, :( ($(Expr(:., outsym, Expr(:quote, :val))), $gsym) ) )

		# enclose in a try block
		body = Expr(:try, Expr(:block, body...),
				          :e, 
				          quote 
				          	if isa(e, OutOfSupportError)
				          		return(-Inf, zero($PARAM_SYM))
				          	else
				          		throw(e)
				          	end
				          end)

	else  # case without gradient
		# body = [ betaAssign(m)...,         # assigments beta vector -> model parameter vars
		#          :($ACC_SYM = LLAcc(0.)),  # initialize accumulator
  #                m.source.args...,         # model statements
  #                :(return $( Expr(:., ACC_SYM, Expr(:quote, :val)) )) ]

		# # enclose in a try block
		# body = Expr(:try, Expr(:block, body...),
		# 		          :e, 
		# 		          :(if isa(e, OutOfSupportError); return(-Inf); else; throw(e); end) )

		# header = Expr[]
	end


	# build and evaluate the let block containing the function and external vars hooks
	fn = newvar(:ll)
	body = Expr(:function, Expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	Expr(:block, body) )
	body = Expr(:let, Expr(:block, :(global $fn), header..., body))

	# println("#############\n$body\n############")

	debug ? body : (eval(body) ; (eval(fn), m.bsize, m.pars, m.init) )
end

#### translates ~ into regular syntax
function translate(ex::Expr)
	if ex.head == :block 
		return Expr(:block, translate(ex.args)...)
	elseif ex.head == :call && ex.args[1] == :~
		fn = symbol("logpdf$(ex.args[3].args[1])")
		return :( $ACC_SYM += logpdf( $(ex.args[3]), $(ex.args[2]) ) )
	else
		return ex
	end
end
translate(ex::Vector) = map(translate, ex)
translate(ex::Any) = ex

# ACC_SYM = :_acc
# ex = quote
# 	b=a+6
# 	z = log(Y)
# 	z ~ Normal(1,2)
# end
# dump(ex)
# typeof(ex)
# typeof(ex.args)
# ex.head
# Expr(:block, translate(ex.args)...)
# ex
# translate(ex)

# translate(:(x ~Bernoulli(1,2)) )
# translate([:(b=a+6), :(x ~Bernoulli(1,2)) ] )
