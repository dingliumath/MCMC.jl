cd("p:/Documents/julia/MCMC.jl/src/autodiff")
cd("~/devl/MCMC.jl/src/autodiff")
pwd()

@windows_only include("p:/Documents/julia/MCMC.jl.fredo/src/autodiff/mymod.jl")
@unix_only include("/home/fredo/devl/MCMC.jl/src/autodiff/mymod.jl")


######

	b = 12
	Abcd.debug(:( a*b ), a=0)
	b = [1, 2, 3]
	Abcd.debug(:( a*b ), a=0)
	f, n, p, w = Abcd.generateModelFunction(:( a*b ), gradient=true, a=0)
	f([0.])
	f, n, p, w = Abcd.generateModelFunction(:( b*a ), gradient=true, a=0)
	f([0.])

	b = [1. 2 3]
	f, n, p, w = Abcd.generateModelFunction(:( a*b ), gradient=true, a=0)
	f([0.])
	f, n, p, w = Abcd.generateModelFunction(:( b*a ), gradient=true, a=0)
	f([0.])
	f, n, p, w = Abcd.generateModelFunction(:( b*a ), gradient=true, a=[0., 1, 2])
	f([0., 1, 2])
	Abcd.debug(:( b*a ), a=[0., 1, 2])
	Abcd.generateModelFunction(:( b*a ), gradient=true, a=[0., 1, 2], debug=true)

	f, a,b,c = Abcd.generateModelFunction(:( x+y ), gradient=true, x=0)
	f, a,b,c = Abcd.generateModelFunction(:( a=x ), gradient=true, x=0)
	f([-13.])

y = 0.5
x = 1.0
ex = :( x ~ Bernoulli(y) )
Abcd.debug(ex, y = 0.5)
f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, y=0.5)
f([0.5])[1] - f([0.5001])[1]

Abcd.debug(ex, y = [0.5, 0.2])
f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, y=[0.5, 0.2])


test{T<:Distribution}(ds::Array{T}) = "ok"
test(ds::Array) = "array"
test2(ds::Array{Distribution}) = "array2"
test3(ds::Array{Distributions.Bernoulli}) = "array2"

test(z)
test2(z)
test3(z)

isa(Distributions.Bernoulli, Distribution)



using Base.LinAlg.BLAS

                a = zeros(3)
                _acc = LLAcc(0.0)
                _d_acc1259 = 1.0
                _da = zeros(Float64,(3,))
                _dtmp1258 = zeros(Float64,(1,))
                tmp1258 = *(b,a)
                _acc1259 = +(_acc,tmp1258)
                for i = 1:length(tmp1258) # line 40:
                    _dtmp1258[i] += _d_acc1259
                end
                gemm!('T','N',1.0,b, _dtmp1258,1.0,_da)
                local _d_beta = similar(_beta)
                _d_beta[1:3] = _da

[_dtmp1258 ]
Abcd.vhint[symbol("##tmp#1250")]
length(Abcd.vhint[symbol("##tmp#1250")])
_da = zeros(Float64,(3,))
gemm!('T','N', 1.0, b, [1], 1.0,_da)



a
b
let
	a = 12
	b = a + 4
	println(b)
end



################

# simulate dataset
	srand(1)
	nbeta = 10 # number of predictors, including intercept
	beta0 = randn((nbeta,))

	n = 1000
	X = [ones(n) randn((n, nbeta-1))]
	Y = float64( rand(n) .< ( 1 ./ (1. + exp(X * beta0))) )

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

Abcd.debug(ex, vars=zeros(nbeta))  
Abcd.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)  
f, t1, t2, t3 = Abcd.generateModelFunction(ex, vars=zeros(nbeta), gradient=true)  
f(t3)


using Distributions

myex = Abcd.d_Bernoulli_x1([Distributions.Bernoulli(x) for x in [0.1,0.2]])


		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		smap[:dacc] = symbol("$DERIV_PREFIX$vs")


ex2 = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = sum(vars) * sum(X)
	prob ~ Normal(1,2)
end
Abcd.debug(ex2, vars=0.0)  # no method Bernoulli(Array{Float64,1},)
