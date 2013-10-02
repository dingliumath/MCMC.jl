##########################################################################################
#
#    Derivation rules & function 'derive' returning the expr of gradient
#
##########################################################################################
# TODO : add operators : hcat, vcat, ? : , map, mapreduce, if else 

#########   function to declare a new type in Autodiff (for extensibility)  ######
linkType(a::Type, na::Symbol) = eval(:( $na = $a ))

#########   macro and function to simplify derivation rules creation  ###########
function dfunc(func::Expr, dv::Symbol, diff::Expr)
	argsn = map(e-> isa(e, Symbol) ? e : e.args[1], func.args[2:end])
	index = find(dv .== argsn)[1]

	# change var names in signature and diff expr to x1, x2, x3, ..
	smap = { argsn[i] => symbol("x$i") for i in 1:length(argsn) }
	# symbols for distributions
	smap[ symbol("d$dv")] = symbol("dacc") 
	smap[ symbol("d$(dv)1")] = symbol("dacc1")  # notation for Distributions fields derivatives
	smap[ symbol("d$(dv)2")] = symbol("dacc2")  # notation for Distributions fields derivatives

	args2 = substSymbols(func.args[2:end], smap)

	# diff function name
	fn = symbol("d_$(func.args[1])_x$index")

	fullf = Expr(:(=), Expr(:call, fn, args2...), Expr(:quote, substSymbols(diff, smap)) )
	eval(fullf)
end

# macro version
macro dfunc(func::Expr, dv::Symbol, diff::Expr)
	dfunc(func, dv, diff)
end

# ##  for logpdf(Distrib,x)
# macro dlogpdfd(dist::Symbol, rule)
# 	sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
# 	dfunc( sig, :d, rule ) 

# 	sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
# 	rule2 = substSymbols(rule, {:x => :(x[i]), :ds => :(ds[i])})
# 	dfunc( sig, :d, :(for i in 1:length(x) ; $rule2 ; end))

# 	sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
# 	rule2 = substSymbols(rule, {:dd1 => :(dd1[i]), :dd2 => :(dd2[i]), :dd3 => :(dd3[i]), 
# 		:x => :(x[i]), :ds => :(ds[i]), :d => :(d[i]) })
# 	dfunc(sig, :d, :(for i in 1:length(x) ; $rule2 ; end))
# end

# macro dlogpdfx(dist::Symbol, rule)
# 	sig = :( logpdf($(Expr(:(::), :d, dist)), x::Real) )
# 	dfunc( sig, :x, rule ) 

# 	sig = :( logpdf($(Expr(:(::), :d, dist)), x::AbstractArray) )
# 	rule2 = substSymbols(rule, {:dx => :(dx[i]), :x => :(x[i]), :ds => :(ds[i])})
# 	dfunc( sig, :x, :(for i in 1:length(x) ; $rule2 ; end))

# 	sig = :( logpdf($(Expr(:(::), :d, Expr(:curly, :Array, dist))), x::AbstractArray) )
# 	rule3 = substSymbols(rule2, {:d => :(d[i])})
# 	dfunc( sig, :x, :(for i in 1:length(x) ; $rule3 ; end))
# end




########   rules definitions   #############

## common operators

# addition
@dfunc +(x::Real , y::Real )    x     dx += ds
@dfunc +(x::Real , y::AbstractArray)    x     for i in 1:length(ds) ; dx += ds[i]     ;end
@dfunc +(x::AbstractArray, y       )    x     for i in 1:length(ds) ; dx[i] += ds[i]  ;end
@dfunc +(x::Real , y::Real )    y     dy += ds
@dfunc +(x::AbstractArray, y::Real )    y     for i in 1:length(ds) ; dy    += ds[i]  ;end
@dfunc +(x       , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += ds[i]  ;end

# log lik accumulator type 
# Note : only additions are possible with LLAcc type 
# @dfunc +(x::LLAcc, y       )    x     dx += ds
# @dfunc +(x::LLAcc, y::Real)     y     dy += ds
# @dfunc +(x::LLAcc, y::AbstractArray)    y     for i in 1:length(y) ; dy[i] += ds ;end

# unary substraction
@dfunc -(x::Real )              x     dx -= ds
@dfunc -(x::AbstractArray)              x     for i in 1:length(ds) ; dx[i] -= ds[i]  ;end

# binary substraction
@dfunc -(x::Real , y::Real )    x     dx += ds
@dfunc -(x::Real , y::AbstractArray)    x     for i in 1:length(ds) ; dx += ds[i]     ;end
@dfunc -(x::AbstractArray, y       )    x     for i in 1:length(ds) ; dx[i] += ds[i]  ;end
@dfunc -(x::Real , y::Real )    y     dy -= ds
@dfunc -(x::AbstractArray, y::Real )    y     for i in 1:length(ds) ; dy    -= ds[i]  ;end
@dfunc -(x       , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] -= ds[i]  ;end

# sum()
@dfunc sum(x::Real )       x     dx += ds
@dfunc sum(x::AbstractArray)       x     for i in 1:length(x) ; dx[i] += ds     ;end

# dot()
@dfunc dot(x::AbstractArray, y::AbstractArray)    x     for i in 1:length(x) ; dx[i] += y[i]*ds ;end
@dfunc dot(x::AbstractArray, y::AbstractArray)    y     for i in 1:length(y) ; dy[i] += x[i]*ds ;end

# log() and exp()
@dfunc log(x::Real )       x     dx += ds / x
@dfunc log(x::AbstractArray)       x     for i in 1:length(ds) ; dx[i] += ds[i] / x[i]  ;end

@dfunc exp(x::Real )       x     dx += exp(x) * ds    # TODO : allow :s placeholder for optimization
@dfunc exp(x::AbstractArray)       x     for i in 1:length(ds) ; dx[i] += exp(x[i]) * ds[i] ;end

# sin() and cos()
@dfunc sin(x::Real )       x     dx += cos(x) * ds
@dfunc sin(x::AbstractArray)       x     for i in 1:length(ds) ; dx[i] += cos(x[i]) * ds[i] ;end

@dfunc cos(x::Real )       x     dx -= sin(x) * ds
@dfunc cos(x::AbstractArray)       x     for i in 1:length(ds) ; dx[i] -= sin(x[i]) * ds[i] ;end

# abs, max(), min()
@dfunc abs(x::Real )       x     dx += sign(x) * ds
@dfunc abs(x::AbstractArray)       x     for i in 1:length(ds) ; dx[i] += sign(x[i]) * ds[i] ;end

@dfunc max(x::Real , 			y::Real )   		x     dx += (x > y) * ds
@dfunc max(x::Real , 			y::AbstractArray)   x     for i in 1:length(ds) ; dx += (x > y[i]) * ds[i] ; end
@dfunc max(x::AbstractArray, 	y::Real )   		x     for i in 1:length(ds) ; dx[i] += (x[i] > y) * ds[i] ; end
@dfunc max(x::AbstractArray, 	y::AbstractArray)	x     for i in 1:length(ds) ; dx[i] += (x[i] > y[i]) * ds[i] ; end
@dfunc max(x::Real , 			y::Real )   		y     dy += (x < y) * ds
@dfunc max(x::Real , 			y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += (x < y[i]) * ds[i] ; end
@dfunc max(x::AbstractArray, 	y::Real )   		y     for i in 1:length(ds) ; dy += (x[i] < y) * ds[i] ; end
@dfunc max(x::AbstractArray, 	y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += (x[i] < y[i]) * ds[i] ; end

@dfunc min(x::Real , y::Real )   x     dx += (x < y) * ds
@dfunc min(x::Real , y::AbstractArray)   x     for i in 1:length(ds) ; dx += (x < y[i]) * ds[i] ; end
@dfunc min(x::AbstractArray, y::Real )   x     for i in 1:length(ds) ; dx[i] += (x[i] < y) * ds[i] ; end
@dfunc min(x::AbstractArray, y::AbstractArray)   x     for i in 1:length(ds) ; dx[i] += (x[i] < y[i]) * ds[i] ; end
@dfunc min(x::Real , y::Real )   y     dy += (x > y) * ds
@dfunc min(x::Real , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += (x > y[i]) * ds[i] ; end
@dfunc min(x::AbstractArray, y::Real )   y     for i in 1:length(ds) ; dy += (x[i] > y) * ds[i] ; end
@dfunc min(x::AbstractArray, y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += (x[i] > y[i]) * ds[i] ; end

# multiplication
@dfunc *(x::Real , y::Real )   x     dx += y * ds
@dfunc *(x::Real , y::AbstractArray)   x     for i in 1:length(ds) ; dx += y[i] * ds[i] ; end
@dfunc *(x::AbstractArray, y::Real )   x     for i in 1:length(ds) ; dx[i] += y * ds[i] ; end
@dfunc *(x::AbstractArray, y::Vector)  x     gemm!('N', 'T', 1., ds, reshape(y, length(y), 1), 1., dx)  # reshape needed 
@dfunc *(x::AbstractArray, y::AbstractArray)   x     gemm!('N', 'T', 1., ds, y, 1., dx)

@dfunc *(x::Real , y::Real )   y     dy += x * ds
@dfunc *(x::Real , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x * ds[i] ; end
@dfunc *(x::AbstractArray, y::Real )   y     for i in 1:length(ds) ; dy += x[i] * ds[i] ; end
@dfunc *(x::AbstractArray, y::Vector)  y     gemm!('T', 'N', 1., x, reshape(ds, length(ds), 1), 1., dy)
@dfunc *(x::AbstractArray, y::AbstractArray)   y     gemm!('T', 'N', 1., x, ds, 1., dy)

# dot multiplication
@dfunc .*(x::Real , y::Real )   x     dx += y .* ds
@dfunc .*(x::Real , y::AbstractArray)   x     for i in 1:length(ds) ; dx += y[i] * ds[i] ; end
@dfunc .*(x::AbstractArray, y::Real )   x     for i in 1:length(ds) ; dx[i] += y * ds[i] ; end
@dfunc .*(x::AbstractArray, y::AbstractArray)   x     for i in 1:length(ds) ; dx[i] += y[i] * ds[i] ; end

@dfunc .*(x::Real , y::Real )   y     dy += x * ds
@dfunc .*(x::Real , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x * ds[i] ; end
@dfunc .*(x::AbstractArray, y::Real )   y     for i in 1:length(ds) ; dy += x[i] * ds[i] ; end
@dfunc .*(x::AbstractArray, y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] += x[i] * ds[i] ; end

# power  (both args reals)
@dfunc ^(x::Real, y::Real)  x     dx += y * x ^ (y-1) * ds
@dfunc ^(x::Real, y::Real)  y     dy += log(x) * x ^ y * ds

# dot power
@dfunc .^(x::Real , y::Real )    x     dx += y * x ^ (y-1) * ds
@dfunc .^(x::Real , y::AbstractArray)    x     for i in 1:length(ds) ; dx += y[i] * x ^ (y[i]-1) * ds[i] ; end
@dfunc .^(x::AbstractArray, y::Real )    x     for i in 1:length(ds) ; dx[i] += y * x[i] ^ (y-1) * ds[i] ; end
@dfunc .^(x::AbstractArray, y::AbstractArray)    x     for i in 1:length(ds) ; dx[i] += y[i] * x[i] ^ (y[i]-1) * ds[i] ; end

@dfunc .^(x::Real , y::Real )    y     dy += log(x) * x ^ y * ds
@dfunc .^(x::AbstractArray, y::Real )    y     for i in 1:length(ds) ; dy += log(x[i]) * x[i] ^ y * ds[i] ; end
@dfunc .^(x::Real , y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += log(x) * x ^ y[i] * ds[i] ; end
@dfunc .^(x::AbstractArray, y::AbstractArray)    y     for i in 1:length(ds) ; dy[i] += log(x[i]) * x[i] ^ y[i] * ds[i] ; end

# division
@dfunc /(x::Real , y::Real )   x     dx += ds / y
@dfunc /(x::Real , y::AbstractArray)   x     for i in 1:length(ds) ; dx += ds[i] / y[i] ; end
@dfunc /(x::AbstractArray, y::Real )   x     for i in 1:length(ds) ; dx[i] += ds[i] / y ; end

@dfunc /(x::Real , y::Real )   y     dy -= x * ds / (y * y)
@dfunc /(x::Real , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x * ds[i] / (y[i]*y[i]) ; end
@dfunc /(x::AbstractArray, y::Real )   y     for i in 1:length(ds) ; dy -= x[i] * ds[i] / (y * y); end

# dot division
@dfunc ./(x::Real , y::Real )   x     dx += ds / y
@dfunc ./(x::Real , y::AbstractArray)   x     for i in 1:length(ds) ; dx += ds[i] / y[i] ; end
@dfunc ./(x::AbstractArray, y::Real )   x     for i in 1:length(ds) ; dx[i] += ds[i] / y ; end
@dfunc ./(x::AbstractArray, y::AbstractArray)   x     for i in 1:length(ds) ; dx[i] += ds[i] / y[i] ; end

@dfunc ./(x::Real , y::Real )   y     dy -= x * ds / (y * y)
@dfunc ./(x::Real , y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x * ds[i] / (y[i]*y[i]) ; end
@dfunc ./(x::AbstractArray, y::Real )   y     for i in 1:length(ds) ; dy -= x[i] * ds[i] / (y * y); end
@dfunc ./(x::AbstractArray, y::AbstractArray)   y     for i in 1:length(ds) ; dy[i] -= x[i] * ds[i] / (y[i] * y[i]); end

@dfunc transpose(x::Real )   x   dx += ds
@dfunc transpose(x::AbstractArray)   x   dx += transpose(ds)


# #  All Distribution types constructors
# for d in [:Bernoulli, :TDist, :Exponential, :Poisson]  
# 	dfunc(:( ($d)(p::Real) ), :p, :( dp = ds1 ))
# 	dfunc(:( ($d)(p::AbstractArray) ), :p, :( copy!(dp, ds1) ))
# end

# for d in [:Normal, :Uniform, :Weibull, :Gamma, :Cauchy, :LogNormal, :Binomial, :Beta]
# 	dfunc(:( ($d)(p1::Real, p2::Real) ),   :p1, :( dp1 = ds1 ) )
# 	dfunc(:( ($d)(p1::Real, p2::Real) ),   :p2, :( dp2 = ds2 ) )
# 	dfunc(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p1, :( copy!(dp1, ds1) ) )
# 	dfunc(:( ($d)(p1::AbstractArray, p2::AbstractArray) ), :p2, :( copy!(dp2, ds2) ) )
# end

# ## Normal distribution
# @unix_only begin
# 	@dlogpdfx Normal dx += (d.μ - x) / (d.σ * d.σ) * ds
# 	@dlogpdfd Normal ( 	dd1 += (x - d.μ) / (d.σ*d.σ) * ds;
# 						dd2 += ((x - d.μ)*(x - d.μ) / (d.σ*d.σ) - 1.) / d.σ * ds )
# end
# @windows_only begin
# 	@dlogpdfx Normal dx += (d.mean - x) / (d.std * d.std) * ds
# 	@dlogpdfd Normal ( 	dd1 += (x - d.mean) / (d.std*d.std) * ds;
# 						dd2 += ((x - d.mean)*(x - d.mean) / (d.std*d.std) - 1.) / d.std * ds )
# end

# ## Uniform distribution
# @dlogpdfx Uniform dx += 0.
# @dlogpdfd Uniform ( dd1 += (d.a <= x <= d.b) / (d.b - d.a) * ds ;
# 					dd2 += (d.a <= x <= d.b) / (d.a - d.b) * ds )

# ## Weibull distribution
# @dlogpdfd Weibull   ( 	dd1 += ((1. - (x/d.scale)^d.shape) * log(x/d.scale) + 1./d.shape) * ds ;
# 						dd2 += ((x/d.scale)^d.shape - 1.) * d.shape/d.scale * ds )
# @dlogpdfx Weibull   dx += ((1. - (x/d.scale)^d.shape) * d.shape - 1.) / x * ds

# ## Beta distribution
# @dlogpdfd Beta   ( dd1 += (digamma(d.alpha+d.beta) - digamma(d.alpha) + log(x)) * ds ;
# 				    dd2 += (digamma(d.alpha+d.beta) - digamma(d.beta) + log(1-x)) * ds )
# @dlogpdfx Beta   dx += ((d.alpha-1) / x - (d.beta-1)/(1-x)) * ds

# ## TDist distribution
# @dlogpdfd TDist   dd1 += ((x*x-1)/(x*x + d.df)+log(d.df/(x*x+d.df))+digamma((d.df+1)/2)-digamma(d.df/2))/2 * ds
# @dlogpdfx TDist   dx += (-(d.df+1)*x / (d.df+x*x)) * ds

# ## Exponential distribution
# @dlogpdfd Exponential   dd1 += (x-d.scale) / (d.scale*d.scale) * ds
# @dlogpdfx Exponential   dx -= ds / d.scale

# ## Gamma distribution
# @dlogpdfd Gamma   ( dd1 += (log(x) - log(d.scale) - digamma(d.shape)) * ds ;
# 					dd2 += ((x - d.scale*d.shape) / (d.scale*d.scale)) * ds )
# @dlogpdfx Gamma   dx += (-( d.scale + x - d.shape*d.scale)/(d.scale*x)) * ds

# ## Cauchy distribution
# @dlogpdfd Cauchy   ( dd1 += (2(x-d.location) / (d.scale*d.scale + (x-d.location)*(x-d.location))) * ds ;
# 					 dd2 += (((x-d.location)*(x-d.location) - d.scale*d.scale) / (d.scale*(d.scale*d.scale + (x-d.location)*(x-d.location)))) * ds )
# @dlogpdfx Cauchy   dx += (2(d.location-x) / (d.scale*d.scale + (x-d.location)*(x-d.location))) * ds

# ## Log-normal distribution
# @dlogpdfd LogNormal   ( dd1 += (log(x) - d.meanlog) / (d.sdlog*d.sdlog) * ds ;
# 					 	dd2 += (d.meanlog*d.meanlog - d.sdlog*d.sdlog - log(x)*(2d.meanlog-log(x))) / (d.sdlog*d.sdlog*d.sdlog) * ds )
# @dlogpdfx LogNormal   dx += (d.meanlog - d.sdlog*d.sdlog - log(x)) / (d.sdlog*d.sdlog*x) * ds 


# # # TODO : find a way to implement multi variate distribs that goes along well with vectorization (Dirichlet, Categorical)
# # # TODO : other continuous distribs ? : Pareto, Rayleigh, Logistic, Levy, Laplace, Dirichlet, FDist
# # # TODO : other discrete distribs ? : NegativeBinomial, DiscreteUniform, HyperGeometric, Geometric, Categorical

# ## Bernoulli distribution (Note : no derivation on x parameter as it is an integer)
# @dlogpdfd Bernoulli     dd1 += 1. / (d.p1 - 1. + x) * ds

# ## Binomial distribution (Note : no derivation on x and n parameters as they are integers)
# @dlogpdfd Binomial      dd2 += (x / d.prob - (d.size-x) / (1 - d.prob)) * ds

# ## Poisson distribution (Note : no derivation on x parameter as it is an integer)
# @dlogpdfd Poisson       dd1 += (x / d.lambda - 1) * ds


