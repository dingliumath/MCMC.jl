#########################################################################
#    testing script for gradients
#########################################################################

using Base.Test
# using MCMC 

@windows_only begin
		include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/mymod.jl")
	end
@unix_only begin
		include("~/devl/MCMC.jl.fredo/src/autodiff/mymod.jl")
	end

testedmodule = Abcd
include("p:/Documents/julia/MCMC.jl.fredo/test/helper_diff.jl")

## variables of different dimension for testing
v0ref = 2.
v1ref = [2., 3, 0.1, 0, -5]
v2ref = [-1. 3 0 ; 0 5 -2]

## regular functions
@test_combin    x+y       size(x)==size(y) || ndims(x)==0 || ndims(y)==0
@test_combin    x+y+z     size(x)==size(y)==size(z) || 
							(ndims(x)==0 && size(y)==size(z)) || 
							(ndims(y)==0 && size(x)==size(z)) ||
							(ndims(z)==0 && size(x)==size(z))
@test_combin    sum(x)
@test_combin    x-y       size(x)==size(y) || ndims(x)==0 || ndims(y)==0
@test_combin    x.*y      size(x)==size(y) || ndims(x)==0 || ndims(y)==0
@test_combin    x./y  	  y->y==0 ? 0.1 : y  size(x)==size(y) || ndims(x)==0 || ndims(y)==0
@test_combin    x.^y      x->x<=0 ? 0.2 : x  size(x)==size(y) || ndims(x)==0 || ndims(y)==0
@test_combin    sin(x)
@test_combin    abs(x)    x->x==0 ? 0.001 : x 
@test_combin    cos(x)
@test_combin    exp(x)
@test_combin    log(x)    x->x<=0 ? 0.1 : x

@test_combin    transpose(x) 
@test_combin    x' 

# @test_combin    max(x,y)  x->x+0.001  # x slightly shifted to avoid numerical derivation fail 
# @test_combin    min(x,y)

@test_combin    x^y       ndims(x)==ndims(y)==0

@test_combin    x/y       y->y==0 ? 0.1 : y ndims(x)==0 || ndims(y)==0

@test_combin    x*y       ndims(x)==0 || ndims(y)==0 || size(x,2)==size(y,1)
tz = transpose(v1ref)
deriv1(:(x*tz), [-3., 2, 0]) 
deriv1(:(tz*x), v1ref)  
deriv1(:(v2ref*x), [-3., 2, 0])
deriv1(:(v2ref[:,1:2]*x), [-3. 2 0 ; 1 1 -2]) 

@test_combin    dot(x,y)  ndims(x)>0 && ndims(y)>0 && size(x)==size(y)


## continuous distributions
@test_combin    logpdf(Normal(mu, sigma), x)      sigma->sigma<=0?0.1:sigma size(mu)==size(sigma) &&
													(ndims(mu)==0 || size(x)==size(mu))

@test_combin    logpdf(Uniform(a, b), x)    a->a-10 b->b+10 size(a)==size(b) && (ndims(a)==0 || size(x)==size(a))

@test_combin    logpdf(Weibull(sh, sc), x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc  x->x<=0?0.1:x size(sh)==size(sc) && 
													(ndims(sh)==0 || size(x)==size(sh))

@test_combin    logpdf(Gamma(sh, sc), x)    sh->sh<=0?0.1:sh  sc->sc<=0?0.1:sc x->x<=0?0.1:x  size(sh)==size(sc) && 
												(ndims(sc)==0 || size(sc)==size(x))

# @mtest testpattern1 logpdfBeta(a,b,x)         x->clamp(x, 0.01, 0.99) a->a<=0?0.1:a b->b<=0?0.1:b
@test_combin    logpdf(Beta(a,b),x)         x->clamp(x, 0.01, 0.99) a->a<=0?0.1:a b->b<=0?0.1:b (size(a)==size(b)) && (ndims(a)==0 || size(x)==size(a))

# @mtest testpattern1 logpdfTDist(df,x)         df->df<=0?0.1:df
@test_combin    logpdf(TDist(df),x)         df->df<=0?0.1:df    (size(df)==size(x)) || ndims(df)==0

@mtest testpattern1 logpdfExponential(sc,x)   sc->sc<=0?0.1:sc  x->x<=0?0.1:x
@mtest testpattern1 logpdfCauchy(mu,sc,x)      sc->sc<=0?0.1:sc
@mtest testpattern1 logpdfLogNormal(lmu,lsc,x)  lsc->lsc<=0?0.1:lsc x->x<=0?0.1:x

## discrete distributions
#  the variable x being an integer should not be derived against

# note for Bernoulli : having prob=1 or 0 is ok but will make the numeric differentiator fail => not tested
@test_combin logpdf(Bernoulli(prob),x) prob prob->clamp(prob, 0.01, 0.99) x->float64(x>0) size(prob)==size(x)||ndims(prob)==0


@mtest testpattern1 logpdfPoisson(l,x)   exceptLast l->l<=0?0.1:l x->iround(abs(x)) 
@mtest testpattern1 logpdfBinomial(n, prob,x)   exceptFirstAndLast prob->clamp(prob, 0.01, 0.99) x->iround(abs(x)) n->iround(abs(n)+10)


#########################################################################
#   misc. tests
#########################################################################

# Parsing should throw an error when model parameter is used as an integer variable
try
	deriv1(:(logpdfBernoulli(1, x)), [0.])
	deriv1(:(logpdfPoisson(1, x)), [0.])
	deriv1(:(logpdfBinomial(3, 0.5, x)), [0.])
	deriv1(:(logpdfBinomial(x, 0.5, 2)), [0.])
	throw("no error !!")
catch e
	assert(e != "no error !!", 
		"parser not throwing error when discrete distribution has a parameter dependant integer argument")
end

##  ref  testing
deriv1(:(x[2]),              v1ref)
deriv1(:(x[2:3]),            v1ref)
deriv1(:(x[2:end]),          v1ref)

deriv1(:(x[2:end]),          v2ref)
deriv1(:(x[2]),              v2ref)
deriv1(:(x[2:4]),            v2ref)
deriv1(:(x[:,2]),            v2ref)
deriv1(:(x[1,:]),            v2ref)
deriv1(:(x[2:end,:]),        v2ref)
deriv1(:(x[:,2:end]),        v2ref)

deriv1(:(x[2]+x[1]),          v2ref)
deriv1(:(log(x[2]^2+x[1]^2)), v2ref)

# fail case when individual elements of an array are set several times
# model = :(x::real(3); y=x; y[2] = x[1] ; y ~ TestDiff())


macro mym(func::Expr, constraints...)
	for c in constraints
		dump(c)
	end
end

@mym logpdfPoisson(l,x) exceptLast l->l<=0?0.1:l x->iround(abs(x)) size(x)==(1,2) & dim(prob)==2 

@mtest testpattern6 logpdf(Bernoulli(prob),x)   prob prob->clamp(prob, 0.01, 0.99) x->float64(x>0) 

