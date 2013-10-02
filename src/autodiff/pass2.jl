######### evaluates once all variables to give type hints for derivation ############
#  most gradient calculation statements depend on the type of variables (Scalar or Array)
#  this is where they are evaluated (with values stored in global Dict 'vhint' )
function preCalculate(m::ParsingStruct)
    global vhint = Dict()

    body = Expr[ [ :( $(p[1]) = $(p[2]) ) for p in zip(m.insyms, m.init)]..., 
                 m.exprs...]
    
    vl = Autodiff.getSymbols(body)  # list of all vars (external, parameters, set by model, and accumulator)
    body = vcat(body, 
    			[ :(vhint[$(Expr(:quote, v))] = $v) for v in vl ])

	# identify external vars and add definitions x = Main.x
	header = [ :( local $v = $(Expr(:., :Main, Expr(:quote, v))) ) for v in external(m)]

	# build and evaluate the let block containing the function and external vars hooks
	try
		eval(Expr(:let, Expr(:block, vcat(header, body)...) ))
	catch e
		error("Model fails to evaluate for initial values given")
	end

	res = vhint[m.outsym]
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

	avars = activeVars(m)
	for ex2 in reverse(m.exprs)  # proceed backwards
		assert(isa(ex2, Expr), "[backwardSweep] not an expression : $ex2")
		explore(ex2)
	end
end



##########################################################################################
#
#    Derivation function 'derive' returning the expr of gradient
#
##########################################################################################

## returns sample value for the given Symbol or Expr (for refs)
hint(v::Symbol) = vhint[v]
hint(v) = v  # should be a value if not a Symbol or an Expression
function hint(v::Expr)
	assert(v.head == :ref, "[hint] unexpected variable $v")
	v.args[1] = :( vhint[$(Expr(:quote, v.args[1]))] )
	eval(v)
end

#########   Returns gradient expression of opex       ###########
function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))  # opex=:(z^x);index=2;dsym=:y
	vs = opex.args[1+index]
	ds = dprefix(dsym)
	args = opex.args[2:end]
	
	val = map(hint, args)  # get sample values of args to find correct gradient statement

	fn = symbol("d_$(opex.args[1])_x$index")

	try
		dexp = eval(Expr(:call, fn, val...))

		smap = { symbol("x$i") => args[i] for i in 1:length(args)}
		smap[:ds] = ds
		smap[:ds1] = symbol("$(ds)#1")
		smap[:ds2] = symbol("$(ds)#2")
		smap[:dacc] = dprefix(vs)
		smap[:dacc1] = dprefix("$(vs)#1")
		smap[:dacc2] = dprefix("$(vs)#2")
		dexp = substSymbols(dexp, smap)

		return dexp
	catch e 
		error("[derive] Failed to derive $opex by argument $vs ($(map(typeof, val)))")
	end

end