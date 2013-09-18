############# Distribution types vectorizations   ################
# single parameter distributions
import Distributions.Bernoulli, 
	Distributions.TDist, 
	Distributions.Exponential, 
	Distributions.Poisson

for d in [:Bernoulli, :TDist, :Exponential, :Poisson]  
	@eval begin
		function ($d)(ps::Array)
			ds = Array($d, size(ps))
			for i in 1:length(ds)
				ds[i] = ($d)(ps[i])
			end
			ds
		end
	end 
end

# two parameter distributions
import Distributions.Normal, 
		Distributions.Uniform, 
		Distributions.Weibull, 
		Distributions.Gamma, 
		Distributions.Cauchy, 
		Distributions.LogNormal, 
		Distributions.Binomial, 
		Distributions.Beta

for d in [:Normal, :Uniform, :Weibull, :Gamma, :Cauchy, :LogNormal, :Binomial, :Beta]
	@eval begin
		function ($d)(p1::Array, p2::Array)
			ds = Array($d, size(p1))
			for i in 1:length(ds)
				ds[i] = ($d)(p1[i], p2[i])
			end
			ds
		end
	end 
end

############# logpdf vectorization on the distribution argument   ################
import Distributions.logpdf
function logpdf{T<:Distribution}(ds::Array{T}, x::AbstractArray)
	res = Array(Float64, size(ds))
	assert( size(ds) == size(x), "x and distributions sizes do not match")
	for i in 1:length(x)
		res[i] = logpdf(ds[i], x[i])
	end
	res
end
