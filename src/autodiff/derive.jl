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


## Returns gradient expression of opex
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