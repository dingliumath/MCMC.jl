using Distributions
using DataFrames

pwd()
cd("MCMC.jl/src")
include("MCMC.jl") 
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


    __beta = zeros(nbeta)
    let 
        global _ll3
        local X = Main.X
        local Y = Main.Y
        function _ll3(_beta::Vector{Float64})
            try 
                vars = _beta[1:10]
                _acc = LLAcc(0.0)
                _d_acc_216 = 1.0
                _dtmp_289 = zeros(Float64,(1000,))
                _dtmp_285 = zeros(Float64,(1000,))
                _dtmp_286 = zeros(Float64,(1000,))
                _dtmp_288_1 = zeros(Float64,(1000,))
                _dtmp_288_2 = zeros(Float64,(1000,))
                _dtmp_284 = zeros(Float64,(10,))
                _d_acc_215 = 0.0
                _dvars = zeros(Float64,(10,))
                _dtmp_287 = zeros(Float64,(1000,))
                _dprob = zeros(Float64,(1000,))
                tmp_283 = Normal(0,1.0)
                tmp_284 = logpdf(tmp_283,vars)
                _acc_215 = +(_acc,tmp_284)
                tmp_285 = *(X,vars)
                tmp_286 = exp(tmp_285)
                tmp_287 = +(1.0,tmp_286)
                prob = /(1,tmp_287)
                tmp_288 = Bernoulli(prob)
                tmp_289 = logpdf(tmp_288,Y)
                _acc_216 = +(_acc_215,tmp_289)
                _d_acc_215 += _d_acc_216
                for i = 1:length(tmp_289) 
                    _dtmp_289[i] += _d_acc_216
                end
                for i = 1:length(_dtmp_289) 
                    _dtmp_288_1[i] += *(/(1.0,+(-(tmp_288[i].p1,1.0),Y[i])),_dtmp_289[i])
                end
                for i = 1:length(_dtmp_288_1)
                    _dprob[i] = _dtmp_288_1[i]
                end
                for i = 1:length(_dprob) 
                    _dtmp_287[i] -= /(*(1,_dprob[i]),*(tmp_287[i],tmp_287[i]))
                end
                for i = 1:length(_dtmp_287)
                    _dtmp_286[i] += _dtmp_287[i]
                end
                for i = 1:length(_dtmp_286) 
                    _dtmp_285[i] += *(exp(tmp_285[i]),_dtmp_286[i])
                end
                gemm!('T','N',1.0,X,reshape(_dtmp_285,length(_dtmp_285),1),1.0,_dvars)
                for i = 1:length(tmp_284) 
                    _dtmp_284[i] += _d_acc_215
                end
                for i = 1:length(_dtmp_284) 
                    _dvars[i] += *(/(-(tmp_283.mean,vars[i]),*(tmp_283.std,tmp_283.std)),_dtmp_284[i])
                end
                local _d_beta = similar(_beta)
                _d_beta[1:10] = _dvars
                (_acc_216.val,_d_beta)
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

_ll1(zeros(nbeta))
@time begin 
		s = 0.
		for i in 1:1000
			t = randn(nbeta)
			s += _ll1(t)[1]
		end
	end  # 0.75 sec




using Base.LinAlg.BLAS
using NumericExtensions

gemm!

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


Bernoulli

using Distributions
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


