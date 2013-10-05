##########################################################################
#
#    Model Expression parsing
#      - transforms MCMC specific idioms into regular Julia syntax
#      - calls Autodiff module for gradient code generation
#      - generates function within MCMC module
#
#    - extends Distributions package to have vectorization on distrib parameter  
#    - sends Autodiff package extra derivation rules for MCMC specific functions and types
#
##########################################################################

module Sandbox

	using Distributions
	using Base.LinAlg.BLAS

    include("autodiff/Autodiff.jl")
	# using Autodiff

	# Distributions extensions, TODO : ask for migration to Distributions package
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

	Autodiff.linkType(LLAcc, :LLAcc)

	include("MCMC_deriv_rules.jl")



end