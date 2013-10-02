##########################################################################
#
#    Module containing the generated log-likelihood functions
#       with necessary definitions
#
#    - extends Distributions package to have vectorization on distrib parameter  
#    - extends Autodiff package for MCMC specific derivation rules and types
#
##########################################################################

module Sandbox

	using Distributions
	using .Autodiff
	
	# Distributions extensions, TODO : ask for migration to Distributions package
	include("vector_dists.jl")

	# naming conventions 
	const ACC_SYM = :_acc       # name of accumulator variable
	const PARAM_SYM = :_beta    # name of parameter vector

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
	+(ll::LLAcc, x::Real) = LLAcc(ll.val + x)
	+(ll::LLAcc, x::Array{Float64}) = LLAcc(ll.val + sum(x))


	include("MCMC_deriv_rules.jl")

end