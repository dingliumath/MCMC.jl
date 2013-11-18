#####################""
    module Mockup3
        type Newtype ; end;

        module Mockup2
            cm = current_module()
            println(" in $cm, parent = $(Base.module_parent(cm))")

            test() = (cm = eval(:(current_module())) ; println(" in $cm, parent = $(Base.module_parent(cm))"))
        end
    end





    Mockup3.Mockup2.test()
    names

    let
        type Abcd ; end;
    end

    Abcd
    Newtype
    Mockup3.Newtype

    Newtype = Mockup3.Newtype

    x = Newtype
    module Mockup
        include("Sandbox.jl")

        # naming conventions 
        const ACC_SYM = :_acc       # name of accumulator variable
        const PARAM_SYM = :_beta    # name of parameter vector
    end

    cd("parser")
    pwd()

    whos()
    Mockup.Autodiff.d_log_x1
    Mockup.Sandbox.Autodiff.d_logpdf_x1

    include("/home/fredo/devl/MCMC.jl.fredo/src/MCMC.jl")
######

	b = 12
	MCMC.Autodiff.debug(:( a*b ), a=0)
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


    include("/home/fredo/devl/MCMC.jl.fredo/src/autodiff/mymod.jl")
    a0=[1.,2] ; ex=:(x~Exponential(a)) ; x = [1.,2]
    Abcd.debug(ex, a=a0)
	f, a,b,c = Abcd.generateModelFunction(ex, a=a0)
    f, a,b,c = Abcd.generateModelFunction(ex, a=a0, gradient=true)
	f([13., 3.])

    y = 0.5
    x = 1.0
    ex = :( x ~ TDist(y) )
    Abcd.debug(ex, y = 0.5)
    Abcd.generateModelFunction(ex, gradient=true, y=0.5, debug=true)
    f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, y=0.5)
    f([0.5])
    (f([0.5001])[1] - f([0.5])[1])*10000

    x = [1.0, 0, 1, 1]
    ex = :( x ~ Bernoulli(y) )
    Abcd.debug(ex, y = 0.5)
    Abcd.generateModelFunction(ex, gradient=true, y=0.5, debug=true)
    f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, y=0.5)
    f([0.5])
    (f([0.5001])[1] - f([0.5])[1])*10000


    x = [1.0, 0, 1, 1]
    ex = :( x ~ Bernoulli(y) )
    Abcd.debug(ex, y = [0.1:0.1:0.4])
    Abcd.generateModelFunction(ex, gradient=true, y=[0.1:0.1:0.4], debug=true)
    f, a,b,c = Abcd.generateModelFunction(ex, gradient=true, y=[0.1:0.1:0.4])
    f([0.1:0.1:0.4])
    (f([0.5001])[1] - f([0.5])[1])*10000

################

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

    module Sandbox 
        @unix_only include("/home/fredo/devl/MCMC.jl.fredo/src/parser/parser.jl")
        @windows_only include("p:/documents/julia/MCMC.jl.fredo/src/parser/parser.jl")
    end
    Sandbox.generateModelFunction(ex, vars=zeros(nbeta), debug=true, gradient=true)
    f,vsize, p, i = Sandbox.generateModelFunction(ex, vars=zeros(nbeta), gradient=true)
    f(i)

    ex2 = quote
        _acc = 0.0
        _acc += sum(logpdf(Normal(0, 1.0), vars))
        prob = 1 / (1. + exp(X * vars)) 
        _acc += sum(logpdf(Bernoulli(prob), Y))
    end

    Sandbox.diff(ex2, :_acc, vars=zeros(nbeta)) 

    m = Sandbox.Autodiff.ParsingStruct()
    m.source = ex2
    Sandbox.Autodiff.unfold!(m)
    m.exprs
    m.insyms = [:vars]
    m.outsym = :_acc
    m.ag, m.dg, subst, m.exprs = Sandbox.Autodiff.varGraph(m.exprs)
    m.ag
    m.dg
    subst
    m.exprs
    m.outsym = symbol("_acc#2#1")

    Sandbox.Autodiff.activeVars(m)
    Sandbox.Autodiff.relations(m.outsym, m.ag)
    Sandbox.Autodiff.relations(m.insyms, m.dg)
    Sandbox.Autodiff.external(m)

    m.init = Any[]
    push!(m.init, zeros(nbeta))
    Sandbox.Autodiff.parent_mod
    Sandbox.Autodiff.preCalculate(m)
    Sandbox.Autodiff.vhint

    Y = 2.
    ex = quote
        res = LLAcc(0.0)
        x = 3+Y
        z = a*x
        res += sum(z)
        res = res.val
    end
    Sandbox.diff(ex , :res, a=2.)


    m = Sandbox.Autodiff.ParsingStruct()
    m.source = ex
    Sandbox.Autodiff.unfold!(m)
    m.exprs
    m.insyms = [:a]
    m.outsym = :res
    m.ag, m.dg, subst, m.exprs = Sandbox.Autodiff.varGraph(m.exprs)

    Sandbox.Autodiff.activeVars(m)
    Sandbox.Autodiff.relations(m.outsym, m.ag)
    Sandbox.Autodiff.relations(m.insyms, m.dg)
    Sandbox.Autodiff.external(m)

    Sandbox.diff(ex , :z, a=2.)

    Abcd.debug(ex, vars=zeros(nbeta))  
    Abcd.generateModelFunction(ex, vars=zeros(nbeta), gradient=true, debug=true)  
    f, t1, t2, t3 = Abcd.generateModelFunction(ex, vars=zeros(nbeta), gradient=true)  
    f(t3)


##########
	let 
        global __ll_351
        local X = Main.X
        local Y = Main.Y
        function __ll_351(_beta::Vector{Float64})
            try 
                vars = _beta[1:10]
                vars = zeros(nbeta)
                _acc = LLAcc(0.0)
                _d___acc_349 = 1.0
                _d__tmp_342 = zeros(Float64,(10,))
                _d__tmp_346 = zeros(Float64,(1000,2))
                _d___acc_348 = 0.0
                _d__tmp_344 = zeros(Float64,(1000,))
                _d__tmp_347 = zeros(Float64,(1000,))
                _d__tmp_345 = zeros(Float64,(1000,))
                _d__tmp_343 = zeros(Float64,(1000,))
                _dvars = zeros(Float64,(10,))
                _dprob = zeros(Float64,(1000,))
                __tmp_341 = Normal(0,1.0)
                __tmp_342 = logpdf(__tmp_341,vars)
                ___acc_348 = +(_acc,__tmp_342)
                __tmp_343 = *(X,vars)
                __tmp_344 = exp(__tmp_343)
                __tmp_345 = +(1.0,__tmp_344)
                prob = /(1,__tmp_345)
                __tmp_346 = Abcd.Bernoulli(prob)
                __tmp_347 = Abcd.logpdf(__tmp_346,Y)
                ___acc_349 = +(___acc_348,__tmp_347)
                _d___acc_348 += _d___acc_349
                for i = 1:length(__tmp_347) 
                    _d__tmp_347[i] += _d___acc_349
                end
                for i = 1:length(_d__tmp_347) 
                    _d__tmp_346[i,1] += *(/(1.0,+(-(__tmp_346[i].p1,1.0),Y[i])),_d__tmp_347[i])
                end
                for i = 1:length(_d__tmp_346)
                    _dprob[i] = _d__tmp_346[i,1]
                end
                for i = 1:length(_dprob)
                    _d__tmp_345[i] -= /(*(1,_dprob[i]),*(__tmp_345[i],__tmp_345[i]))
                end
                for i = 1:length(_d__tmp_345)
                    _d__tmp_344[i] += _d__tmp_345[i]
                end
                for i = 1:length(_d__tmp_344) 
                    _d__tmp_343[i] += *(exp(__tmp_343[i]),_d__tmp_344[i])
                end
                gemm!('T','N',1.0,X,reshape(_d__tmp_343,length(_d__tmp_343),1),1.0,_dvars)
                for i = 1:length(__tmp_342) 
                    _d__tmp_342[i] += _d___acc_348
                end
                for i = 1:length(_d__tmp_342)
                    _dvars[i] += *(/(-(__tmp_341.mean,vars[i]),*(__tmp_341.std,__tmp_341.std)),_d__tmp_342[i])
                end
                local _d_beta = similar(_beta)
                _d_beta[1:10] = _dvars
                (___acc_349.val,_d_beta)
            catch e 
                if isa(e,OutOfSupportError)
                    return (-(Inf),zero(_beta))
                else  
                    throw(e)
                end
            end
        end
    end





d = { :a => 1, :b => 2, :c => 2}
find(collect(values(d)) .== 1)
lter((k,v)-> )
collect(d)
for (k,v) in d ; findfirst

get(d, :b, nothing)
getkey(d, :b, nothing)
indexin(values)
collect(keys(d))[findin(collect(values(d)), [3])[1]]

d
filter((k,v)-> v==2, d)
map((k,v)->)


A = [1. 2 ; 3 4]
B = ones(2,2)
A * B
using Base.LinAlg
A_mul_B(A,B)

C = similar(A)
A_mul_Bt(C,A,B)
C

Bv = [4. 5]
A_mul_Bt(C,A,Bv)


ex = :(δa = 12 )
ex.args[1] == :da

ɑ
α=4
using Distributions
d = Normal(1,2)
d.μ
μ = 3
names(d)
μ=2

a+2
δtmp = fill!(δtmp, 0.)

