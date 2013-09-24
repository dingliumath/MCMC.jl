######### evaluates once all variables to give type hints for derivation ############
#  most gradient calculation statements depend on the type of variables (Scalar or Array)
#  this is where they are evaluated (with values stored in global Dict 'vhint' )
function preCalculate(m::ParsingStruct)
    global vhint = Dict()

    body = Expr[ betaAssign(m)..., 
                 :($ACC_SYM = LLAcc(0.)), 
                 m.exprs...]
    
    vl = getSymbols(body)  # list of all vars (external, parameters, set by model, and accumulator)
    body = vcat(body, 
    			[ :(vhint[$(Expr(:quote, v))] = $v) for v in vl ], 
    			:(return $(Expr(:., m.finalacc, Expr(:quote, :val)))) )

	# enclose in a try block to catch zero likelihoods (-Inf log likelihood)
	body = Expr(:try, Expr(:block, body...),
			          :e, 
			          Expr(:block, :(if isa(e, OutOfSupportError); return(-Inf); else; throw(e); end)))

	# identify external vars and add definitions x = Main.x
	ev = setdiff(m.accanc, union(m.varsset, Set(ACC_SYM, collect(keys(m.pars))...))) # vars that are external to the model
	vhooks = Expr(:block, [ :( local $v = $(Expr(:., :Main, Expr(:quote, v))) ) for v in ev]...) # assigment block

	# build and evaluate the let block containing the function and external vars hooks
	fn = newvar()
	body = Expr(:function, Expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	Expr(:block, body) )
	body = :(let; global $fn; $vhooks; $body; end)
	
	eval(body)
	fn = eval(fn)

	# now evaluate vhint (or throw error if model does not evaluate for given initial values)
	res = fn(m.init)
	!isa(res, Real) && error("Model outcome should be a scalar, $(typeof(res)) found")
	res == -Inf && error("Initial values out of model support, try other values")
end


######### builds the gradient expression from unfolded expression ##############
function backwardSweep!(m::ParsingStruct)  

	explore(ex::Expr) = explore(toExprH(ex))
	explore(ex::ExprH) = error("[backwardSweep] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) = nothing

	function explore(ex::Exprequal)
		lhs = ex.args[1]
		assert(isSymbol(lhs) || isRef(lhs), "[backwardSweep] not a symbol / ref on LHS of assigment $(ex)")
		dsym = lhs
		dsym2 = dprefix(lhs)
		
		rhs = ex.args[2]
		if !isSymbol(rhs) && !isa(rhs,Expr) # some kind of number, nothing to do

		elseif isSymbol(rhs) 
			if contains(avars, rhs)
				vsym2 = dprefix(rhs)
				push!(m.dexprs, :( $vsym2 = $dsym2 ))
			end

		elseif isRef(rhs)
			if contains(avars, rhs.args[1])
				vsym2 = dprefix(rhs)
				push!(m.dexprs, :( $vsym2 = $dsym2))
			end

		elseif isa(toExprH(rhs), Exprcall)  
			for i in 2:length(rhs.args) 
				vsym = rhs.args[i]
				if isa(vsym, Symbol) && contains(avars, vsym)
					m.dexprs = vcat(m.dexprs, derive(rhs, i-1, dsym))
				end
			end
		else 
			error("[backwardSweep] can't derive $rhs")
		end
	end

	avars = intersect(m.accanc, m.pardesc)
	for ex2 in reverse(m.exprs)  # proceed backwards
		assert(isa(ex2, Expr), "[backwardSweep] not an expression : $ex2")
		explore(ex2)
	end
end
