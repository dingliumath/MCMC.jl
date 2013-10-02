
@windows_only include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/Autodiff.jl")
@unix_only include("/home/fredo/devl/MCMC.jl.fredo/src/autodiff/Autodiff.jl")

#############################
	m = Autodiff.ParsingStruct()
	m.source = :( a = 45 ; b += 45; copy!(b,a); c = X')
	Autodiff.unfold!(m)
	m.exprs
#############################
	ex = quote
		a = b+56
		a = a + exp(X)
		res = sum(a)
	end

	X = [1. 2 ; 0  4]
	res = Autodiff.diff(ex, :res, b=12)

	Autodiff.diff(:( copy!(a,b) ), :a, b=1)
#############################
	ex = quote
		a = b+56
		copy!(b,a)
		res = sum(a)
	end

	res = Autodiff.diff(ex, :res, b=12)
	m = Autodiff.ParsingStruct()
	m.source = ex
	m.insyms = [:b]
	m.outsym= :res
	Autodiff.unfold!(m)
	m.exprs
#############################
	m = Autodiff.ParsingStruct()
	m.exprs = [:(a=b+3), :(c=a*2), :(c=a*3), :(d=a+c)]
	Autodiff.varGraph!(m)
	m.exprs = [:(a=b+3), :(b=c*2)]
	Autodiff.varGraph!(m)

	m.exprs = [:(a=b+3), :(c=a*2), :(d=log(c)), :(c=c+5)]
	Autodiff.varGraph!(m)
	m.exprs = [:(a=b+3), :(c=a*2), :(d=log(c)), :(c=c+5), :(e=sin(c))]
	Autodiff.varGraph!(m)

	Autodiff.varGraph([:(a=b+3), :(copy!(c,a)), :(c=c+5), :(e=sin(c))])
	Autodiff.varGraph([:(a=b+3), :(c=3), :(c=c+5), :(copy!(c,a)), :(e=sin(c))])
	Autodiff.varGraph([:(a=b+3), :(c=a+5), :(copy!(c,a)), :(e=sin(c))])

	m.exprs = [:(a=b+3), :(c=3), :(c=c+5), :(copy!(c,a)), :(e=sin(c))]
	Autodiff.varGraph!(m)
#############################

	srand(1)
	n = 1000
	nbeta = 10 # number of covariates, including intercept
	X = [ones(n) randn((n, nbeta-1))]  # covariates
	beta0 = randn((nbeta,))
	Y = rand(n) .< ( 1 ./ (1. + exp(-X * beta0)))

	model = quote
			acc = 0.0
			acc += sum( logpdf(Normal(0,1), vars) )
		    prob = 1 / (1. + exp(-X * vars)) 
		    acc += sum( logpdf(Bernoulli(prob), Y))
	end

	Autodiff.diff(model, :acc, vars=zeros(nbeta))
	
	m = Autodiff.ParsingStruct()
	m.outsym = :acc
	m.insyms = [:vars]
	m.init = zeros(5)
	m.source = model
	m.exprs = Expr[]

	Autodiff.unfold!(m)	
	m.exprs
	m.ag, m.dg, subst, m.exprs = Autodiff.varGraph(m.exprs)
	m.exprs
	follow(s) = haskey(subst,s) ? follow(subst[s]) : s
	m.outsym = follow(m.outsym)
	m.ag
	Autodiff.relations(m.outsym, m.ag)
	ui = setdiff(Set(m.insyms...), Autodiff.relations(m.outsym, m.ag))
	ui != Set() && error("some input variables ($ui) do not influence outcome")

	Autodiff.relations(m.insyms, m.dg)
	Autodiff.relations(:Y, m.dg)

	m.ag, m.dg, m.exprs = varGraph(m.exprs)



	Autodiff.diff(:( copy!(a,b) ; acc = a^x ), :acc, b=1)

	explore(v::Symbol) = union(Set(v), haskey(vg, v) ? mapreduce(explore, union, vg[v]) : Set())
vg = [ :c => Set{Symbol}(:a),
	symbol("c#4") => Set{Symbol}(:a),
	:e => Set{Symbol}(symbol("c#4")),
	:a => Set{Symbol}(:b)] 

	ancestors(v::Symbol) = union(Set(v), haskey(vg, v) ? mapreduce(ancestors, union, vg[v]) : Set())
	ancestors(v::Symbol) = haskey(vg, v) ? union(vg[v], map(ancestors, [vg[v]...])...) : Set()
ancestors(:e)

explore(:e) 
haskey(vg, :e)

######################
	srand(1)
	n = 1000
	nbeta = 10 # number of covariates, including intercept

	X = [ones(n) randn((n, nbeta-1))]  # covariates

	beta0 = randn((nbeta,))
	Y = rand(n) .< ( 1 ./ (1. + exp(-X * beta0)))

	# define model
	ex = quote
	    acc = sum( logpdf(Normal(0,1), vars) )
	    prob = 1 / (1. + exp(-X * vars)) 
	    acc = acc + sum(logpdf(Bernoulli(prob), Y))
	end

head, body, rsym = Autodiff.diff(ex, :acc, vars=zeros(nbeta))

using Distributions
using Base.LinAlg.BLAS

Distributions.Normal(1,1)

include("myf.jl")

myf(ones(nbeta))
myf10000() = (for i in 1:1000 ; myf(ones(nbeta)) ; end )

using Profile

myf10000()
sprofile_clear()
@sprofile myf10000()
sprofile_tree()






