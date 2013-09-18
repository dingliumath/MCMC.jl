using Distributions
using DataFrames

pwd()
cd("MCMC.jl/src")
include("p:/documents/julia/MCMC.jl/src/MCMC.jl") 
# using MCMC

# simulate dataset
srand(1)
nbeta = 10 # number of predictors, including intercept
beta0 = randn((nbeta,))

n = 1000
X = [ones(n) randn((n, nbeta-1))]
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
ex = quote
	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

MCMC.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)

#####  version ref #####
	let 
        global _ll1
        local X = Main.X
        local Y = Main.Y
        local __tmp_258 = -(1.0,Y)
        local __tmp_261 = -(1)
        local __tmp_265 = transpose(X)
        local __tmp_267 = .*(1.0,1.0)
        function _ll1(__beta::Vector{Float64})
            try 
                local vars = __beta[1:10]
                local __acc = 0.0
                local d____acc_256 = 1.0
                local d__tmp_250 = 0.0
                local d__tmp_251 = zeros(Float64,(n,))
                local d__tmp_253 = zeros(Float64,(n,))
                local d__tmp_254 = 0.0
                local d__tmp_252 = zeros(Float64,(n,))
                local dvars = zeros(Float64,(10,))
                local dprob = zeros(Float64,(n,))
                local d____acc_255 = 0.0
                local __tmp_250 = MCMC.logpdfNormal(0,1.0,vars)
                local ____acc_255 = +(__acc,__tmp_250)
                local __tmp_251 = *(X,vars)
                local __tmp_252 = exp(__tmp_251)
                local __tmp_253 = +(1.0,__tmp_252)
                local prob = /(1,__tmp_253)
                local __tmp_254 = MCMC.logpdfBernoulli(prob,Y)
                local ____acc_256 = +(____acc_255,__tmp_254)
                d____acc_255 = +(d____acc_255,sum(d____acc_256))
                d__tmp_254 = +(d__tmp_254,sum(d____acc_256))
                local __tmp_259 = -(prob,__tmp_258)
                local __tmp_260 = ./(1.0,__tmp_259)
                dprob = +(dprob,*(__tmp_260,d__tmp_254))
                local __tmp_262 = .*(__tmp_253,__tmp_253)
                local __tmp_263 = ./(__tmp_261,__tmp_262)
                d__tmp_253 = +(d__tmp_253,.*(__tmp_263,dprob))
                d__tmp_252 = +(d__tmp_252,+(d__tmp_253))
                local __tmp_264 = exp(__tmp_251)
                d__tmp_251 = +(d__tmp_251,.*(__tmp_264,d__tmp_252))
                dvars = +(dvars,*(__tmp_265,d__tmp_251))
                d__tmp_250 = +(d__tmp_250,sum(d____acc_255))
                local __tmp_266 = -(0,vars)
                local __tmp_268 = ./(__tmp_266,__tmp_267)
                dvars = +(dvars,*(__tmp_268,d__tmp_250))
                local d__beta = similar(__beta)
                d__beta[1:10] = dvars
                (____acc_256,d__beta)
            catch e
                if (e=="give up eval") 
                    return (-(Inf),zero(__beta))
                else 
                    throw(e)
                end
            end
        end
    end


#####  version optim #####
	let 
        global _ll2
        local X = Main.X
        local Y = Main.Y
        local __tmp_258 = -(1.0,Y)
        local __tmp_261 = -(1)
        local __tmp_265 = transpose(X)
        local __tmp_267 = .*(1.0,1.0)
 
        local d__tmp_251 = Array(Float64,(n,))
        local d__tmp_253 = Array(Float64,(n,))
        local d__tmp_252 = Array(Float64,(n,))
        local dvars = Array(Float64,(10,))
        local dprob = Array(Float64,(n,))
        local prob = Array(Float64,(n,))
        local __tmp_251 = Array(Float64,(n,))
        local __tmp_253 = Array(Float64,(n,))
        local __tmp_262 = Array(Float64,(n,))
        local __tmp_260 = Array(Float64,(n,))
        local __tmp_259 = Array(Float64,(n,))
        local __tmp_263 = Array(Float64,(n,))

        function _ll2(__beta::Vector{Float64})
            try 
                local vars = __beta[1:10]
                local __acc = 0.0
                local d____acc_256 = 1.0
                local d__tmp_250 = 0.0
                local d__tmp_254 = 0.0
                local d____acc_255 = 0.0

                fill!(d__tmp_251, 0.)
                fill!(d__tmp_253, 0.)
                fill!(d__tmp_252, 0.)
                fill!(dvars, 0.)
                fill!(dprob, 0.)

                local __tmp_250 = MCMC.logpdfNormal(0,1.0,vars)
                local ____acc_255 = +(__acc,__tmp_250) #  scalar only
                
                gemm!('N', 'N', 1., X, reshape(vars,10,1), 0., __tmp_251) #   = *(X,vars)
                
                exp!(__tmp_251)
                map!(Add(), __tmp_253, 1., __tmp_251)
                map!(Divide(), prob, 1., __tmp_253)
                local __tmp_254 = MCMC.logpdfBernoulli(prob,Y)
                local ____acc_256 = +(____acc_255,__tmp_254) #  scalar only

                d____acc_255 = +(d____acc_255, d____acc_256) #  scalar only
                d__tmp_254 = +(d__tmp_254, d____acc_256)   #  scalar only

                # local __tmp_259 = -(prob,__tmp_258)
                map!(Subtract(), __tmp_259, prob,__tmp_258)
                # local __tmp_260 = ./(1.0,__tmp_259)
                # dprob = +(dprob,*(__tmp_260,d__tmp_254))
                map!(Divide(), __tmp_260, d__tmp_254, __tmp_259)
                map1!(Add(), dprob, __tmp_260)

                # local __tmp_262 = .*(__tmp_253,__tmp_253)
                map!(Multiply(), __tmp_262, __tmp_253, __tmp_253)
                # local __tmp_263 = ./(__tmp_261,__tmp_262)
                map!(Divide(), __tmp_263, __tmp_261, __tmp_262)

                # d__tmp_253 = +(d__tmp_253,.*(__tmp_263,dprob))
                map1!(Multiply(), __tmp_263, dprob)
                map1!(Add(), d__tmp_253, __tmp_263)

                # d__tmp_252 = +(d__tmp_252,+(d__tmp_253))
                map1!(Add(), d__tmp_252, d__tmp_253)

                #local __tmp_264 = exp(__tmp_251)   # doublon

                # d__tmp_251 = +(d__tmp_251,.*(__tmp_251,d__tmp_252))  
                map1!(Multiply(), __tmp_251, d__tmp_252)
                map1!(Add(), d__tmp_251, __tmp_251)

                dvars = +(dvars,*(__tmp_265,d__tmp_251))

                d__tmp_250 = +(d__tmp_250, d____acc_255)

                local __tmp_266 = -(0,vars)
                divide!(__tmp_266,__tmp_267)
                dvars = +(dvars,*(__tmp_266,d__tmp_250))

                local d__beta = similar(__beta)
                d__beta[1:10] = dvars
                (____acc_256,d__beta)
            catch e
                if (e=="give up eval") 
                    return (-(Inf),zero(__beta))
                else 
                    throw(e)
                end
            end
        end
    end


#####  version générée  defs  ###
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

    import Distributions.logpdf
    function logpdf{T<:Distribution}(ds::Array{T}, x::Union(Array, BitArray))
    	res = Array(Float64, size(ds))
    	assert( size(ds) == size(x), "x and distributions sizes do not match")
    	for i in 1:length(x)
    		res[i] = logpdf(ds[i], x[i])
    	end
    	res
    end

    import Distributions.Bernoulli
    Bernoulli(ps::Array{Float64}) = map(Bernoulli, ps)

#####  version générée  defs  ###
Y = float64(Y)
Abcd.generateModelFunction(ex, gradient=true, vars=zeros(nbeta), debug=true)
_ll4, s, p, i = Abcd.generateModelFunction(ex, gradient=true, vars=zeros(nbeta))
_ll4(zeros(nbeta))

    __beta = zeros(nbeta)
    let 
        global ll_212
        local X = Main.X
        local Y = Main.Y
        function ll_212(_beta::Vector{Float64})
            try 
                vars = _beta[1:10]
                _acc = LLAcc(0.0)
                _d_acc_226 = 1.0
                _dtmp_320 = zeros(Float64,(1000,))
                _dtmp_321 = zeros(Float64,(1000,))
                _dtmp_324 = zeros(Float64,(1000,))
                _dtmp_319 = zeros(Float64,(10,))
                _d_acc_225 = 0.0
                _dtmp_322 = zeros(Float64,(1000,))
                _dvars = zeros(Float64,(10,))
                _dprob = zeros(Float64,(1000,))
                _dtmp_323_1 = zeros(Float64,(1000,))
                _dtmp_323_2 = zeros(Float64,(1000,))
                tmp_318 = Normal(0,1.0)
                tmp_319 = logpdf(tmp_318,vars)
                _acc_225 = +(_acc,tmp_319)
                tmp_320 = *(X,vars)
                tmp_321 = exp(tmp_320)
                tmp_322 = +(1.0,tmp_321)
                prob = /(1,tmp_322)
                tmp_323 = Bernoulli(prob)
                tmp_324 = logpdf(tmp_323,Y)
                _acc_226 = +(_acc_225,tmp_324)
                _d_acc_225 += _d_acc_226
                for i = 1:length(tmp_324) 
                    _dtmp_324[i] += _d_acc_226
                end
                for i = 1:length(_dtmp_324) 
                    _dtmp_323_1[i] += *(/(1.0,+(-(tmp_323[i].p1,1.0),Y[i])),_dtmp_324[i])
                end
                for i = 1:length(_dtmp_323_1) 
                    _dprob[i] = _dtmp_323_1[i]
                end
                for i = 1:length(_dprob) 
                    _dtmp_322[i] -= /(*(1,_dprob[i]),*(tmp_322[i],tmp_322[i]))
                end
                for i = 1:length(_dtmp_322) 
                    _dtmp_321[i] += _dtmp_322[i]
                end
                for i = 1:length(_dtmp_321)
                    _dtmp_320[i] += *(exp(tmp_320[i]),_dtmp_321[i])
                end
                gemm!('T','N',1.0,X,reshape(_dtmp_320,length(_dtmp_320),1),1.0,_dvars)
                for i = 1:length(tmp_319) 
                    _dtmp_319[i] += _d_acc_225
                end
                for i = 1:length(_dtmp_319) 
                    _dvars[i] += *(/(-(tmp_318.mean,vars[i]),*(tmp_318.std,tmp_318.std)),_dtmp_319[i])
                end
                local _d_beta = similar(_beta)
                _d_beta[1:10] = _dvars
                (_acc_226.val,_d_beta)
            catch e 
                if isa(e,OutOfSupportError) 
                    return (-(Inf),zero(_beta))
                else 
                    throw(e)
                end
            end
        end
    end

##########

module Mybench
    include("p:/documents/julia/MCMC.jl.fredo/src/MCMC.jl")
    include("p:/documents/julia/MCMC.jl.fredo/src/autodiff/mymod.jl")

    srand(1)
    nbeta = 10 # number of predictors, including intercept
    beta0 = randn((nbeta,))

    n = 1000
    X = [ones(n) randn((n, nbeta-1))]
    Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

    # define model
    ex = quote
        vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
        prob = 1 / (1. + exp(X * vars)) 
        Y ~ Bernoulli(prob)
    end

    test0() = (s = 0. ; for i in 1:100 ; t = randn(nbeta) ; s += 1. ; s ; end)

    _ll1, s, p, i = MCMC.generateModelFunction(ex, gradient=true, vars=zeros(nbeta))
    _ll1(zeros(nbeta))
    test1() = (s = 0. ; for i in 1:100 ; t = randn(nbeta) ; s += _ll1(t)[1] ; s ; end)

    _ll4, s, p, i = Abcd.generateModelFunction(ex, gradient=true, vars=zeros(nbeta))
    _ll4(zeros(nbeta))
    test4() = (s = 0. ; for i in 1:100 ; t = randn(nbeta) ; s += _ll4(t)[1] ; s ; end)
end

@time Mybench.test0()  # < 0.0001 sec
@time Mybench.test1()  # 0.105 - 0.160 sec
@time Mybench.test4()  # 0.330 - 0.390 sec


loop1() = (local YL = copy(Main.Y) ; s=0 ; for i in 1:1000; s += sum(Y); end; s)
@time loop1()  # 0.0017
loop2() = (local s=0. ; for i in 1:1000; for j in 1:length(Y) ; s += Y[j]; end ; end; s)
@time loop2()  # 0.24 - 0.30

using Base.LinAlg.BLAS

_ll2(zeros(nbeta))
function f2()
	s = 0.
	for i in 1:1000
		t = randn(nbeta)
		s += _ll2(t)[1]
	end
	s
end
@time f2() ;  # 0.22 sec


_ll3(zeros(nbeta))
function f3()
	s = 0.
	for i in 1:1000
		t = randn(nbeta)
		s += __ll3(t)[1]
	end
	s
end
@time f3() ; # 4.5 sec !!!

##############  avec n = 100_000  #####################

n = 100000
X = [ones(n) randn((n, nbeta-1))]
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))


_ll1(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:10
			t = randn(nbeta)
			s += _ll1(t)[1]
		end
	end  # 0.62 sec


_ll2(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:10
			t = randn(nbeta)
			s += _ll2(t)[1]
		end
	end  # 0.23 sec



##########################################

A = randn(3,2)
ex = quote sum(A*x) end
MCMC.generateModelFunction(ex, debug=true, x=zeros(2), gradient=true)


