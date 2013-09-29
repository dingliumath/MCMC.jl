let
    local Y = Main.Y
    local X = Main.X
    local _dprob = Array(Float64,(1000,))
    local _dtmp_15 = Array(Float64,(1000,))
    local _dtmp_13 = Array(Float64,(1000,))
    local _dtmp_16_1 = Array(Float64,(1000,))
    local _dtmp_16_2 = Array(Float64,(1000,))
    local _dtmp_17 = Array(Float64,(1000,))
    local _dtmp_14 = Array(Float64,(1000,))
    local _dvars = Array(Float64,(10,))
    local _dtmp_11 = Array(Float64,(10,))

    global myf
    function myf(vars)    
        tmp_10 = Distributions.Normal(0,1)
        tmp_11 = Distributions.logpdf(tmp_10,vars)
        acc = sum(tmp_11)
        tmp_12 = -(X)
        tmp_13 = *(tmp_12,vars)  # 444
        tmp_14 = exp(tmp_13)
        tmp_15 = +(1.0,tmp_14)
        prob = /(1,tmp_15)
        tmp_16 = Distributions.Bernoulli(prob)
        tmp_17 = Distributions.logpdf(tmp_16,Y)
        tmp_18 = sum(tmp_17)
        acc_2 = +(acc,tmp_18)
        fill!(_dprob,0.0)
        fill!(_dtmp_15,0.0)
        fill!(_dtmp_13,0.0)
        _dacc_2 = 1.0
        fill!(_dtmp_16_1,0.0)
        fill!(_dtmp_16_2,0.0)
        fill!(_dtmp_17,0.0)   # 8
        fill!(_dtmp_14,0.0)
        _dtmp_18 = 0.0
        _dacc = 0.0
        fill!(_dvars,0.0)
        fill!(_dtmp_11,0.0)
        _dacc += _dacc_2
        _dtmp_18 += _dacc_2
        for i = 1:length(tmp_17) 
            _dtmp_17[i] += _dtmp_18   # 9
        end
        for i = 1:length(Y) 
            _dtmp_16_1[i] += *(/(1.0,+(-(tmp_16[i].p1,1.0),Y[i])),_dtmp_17[i])   # 399
        end
        copy!(_dprob,_dtmp_16_1)
        for i = 1:length(_dprob) 
            _dtmp_15[i] -= /(*(1,_dprob[i]),*(tmp_15[i],tmp_15[i]))  # 327
        end
        for i = 1:length(_dtmp_15)
            _dtmp_14[i] += _dtmp_15[i]
        end
        for i = 1:length(_dtmp_14) 
            _dtmp_13[i] += *(exp(tmp_13[i]),_dtmp_14[i])  # 237
        end
        Base.LinAlg.BLAS.gemm!('T','N',1.0,tmp_12,reshape(_dtmp_13,length(_dtmp_13),1),1.0,_dvars)
        for i = 1:length(tmp_11) 
            _dtmp_11[i] += _dacc
        end
        for i = 1:length(vars)
            _dvars[i] += *(/(-(tmp_10.μ,vars[i]),*(tmp_10.σ,tmp_10.σ)),_dtmp_11[i])
        end

        return (acc_2, _dvars)
    end
end