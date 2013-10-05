#######################################################################
#
#   - builds the model function expression, optionnally with gradient
#   - evaluates the model function expression in Sandbox module
#
#######################################################################

include("sandbox/Sandbox.jl")

function generateModelFunction(model::Expr; gradient=false, debug=false, init...)
	# - 'init' contains the dictionary of model params and their initial value
	# - If 'debug' is set to true, the function returns only the model function 
	#  that would have been created
	
	nmod = Sandbox.prepare(model, init)
	head, body, outsym = Sandbox.Autodiff.diff(nmod; init...)

	m = ParsingStruct()

	## checks initial values
	setInit!(m, init)
	
	## rewrites ~ , do some formatting ... on the model expression
	parseModel!(m, model)

	## process model
	unfold!(m)
	uniqueVars!(m)
	categorizeVars!(m)

	## build function expression
	if gradient  # case with gradient
		preCalculate(m)
		backwardSweep!(m)

		body = Expr[] # list of = expr making the model
		header = Expr[]  # list of expr making the let block var declarations

		# initialization statements 
		body = [ betaAssign(m)...,              # assigments beta vector -> model parameter vars
		         :($ACC_SYM = LLAcc(0.)),       # initialize accumulator
		         :($(dprefix(m.finalacc)) = 1.0)]  # initialize accumulator gradient accumulator  

		avars = setdiff(intersect(m.accanc, m.pardesc), Set(m.finalacc)) # active vars without accumulator, treated above  
		for v in avars 
			vh = vhint[v]
			dsym = dprefix(v)
			if isa(vh, Real)
				push!(body, :($dsym = 0.) )
			elseif 	isa(vh, LLAcc)
				push!(body, :($dsym = 0.) )
			elseif 	isa(vh, Array{Float64})
				push!(header, :( local $dsym = Array(Float64, $(Expr(:tuple,size(vh)...)))) )
				push!(body, :( fill!($dsym, 0.) ) )
			elseif 	isa(vh, Distribution)  #  TODO : find real equivalent vector size
				push!(body, :( $(symbol("$dsym#1")) = 0. ) )
				push!(body, :( $(symbol("$dsym#2")) = 0. ) )
			elseif 	isa(vh, Array) && isa(vh[1], Distribution)  #  TODO : find real equivalent vector size
				push!(header, :( local $(symbol("$dsym#1")) = Array(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
				push!(header, :( local $(symbol("$dsym#2")) = Array(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
				push!(body, :( fill!($(symbol("$dsym#1")), 0.) ) )
				push!(body, :( fill!($(symbol("$dsym#2")), 0.) ) )
			else
				error("[diff] invalid gradient var type $v $(typeof(vh))")
			end
		end

		body = [ body, m.exprs..., m.dexprs...]
		# build function statements, and move to let block constant statements for optimization
		# fvars = union(Set([e.args[1] for e in body]...), Set(PARAM_SYM)) # vars that are re-evaluated at each function call
		# for ex in [m.exprs..., m.dexprs...]
			# if length(intersect(getSymbols(ex.args[2]), fvars)) > 0
				# push!(body, ex)
				# fvars = union(fvars, getSymbols(ex.args[1]))
			# else
			# 	push!(header, ex)
			# end
		# end

		# prefix statements with 'local' at first occurence
		# vars  = Set(PARAM_SYM)
		# for i in 1:length(header)
		# 	if length(intersect(getSymbols(header[i].args[1]), vars)) == 0
		# 		header[i] = :(local $(header[i]))
		# 		vars = union(vars, getSymbols(header[i].args[1]))
		# 	end
		# end
		# for i in 1:length(body)
		# 	if length(intersect(getSymbols(body[i].args[1]), vars)) == 0
		# 		body[i] = :(local $(body[i]))
		# 		vars = union(vars, getSymbols(body[i].args[1]))
		# 	end
		# end

		# return statement (note : gradient vec should match beta variable mapping)
		# dexp = { :( vec([$(dsym(p.sym))]) ) for p in m.pars}
		gsym = dprefix(PARAM_SYM)
		push!(body, :( local $gsym = similar($PARAM_SYM)))
		for p in keys(m.pars)
			v = m.pars[p]
			dsymp = dprefix(p)

			if length(v.dims) == 0  # scalar
				push!(body, :( $gsym[ $(v.pos) ] = $dsymp ) )
			elseif length(v.dims) == 1  # vector
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(body, :( $gsym[ $(Expr(:quote,r)) ] = $dsymp ) )
			else # matrix case  (needs a reshape)
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(body, :( $gsym[ $(Expr(:quote,r)) ] = vec($dsymp) ))
			end
		end
		push!(body, :( ($(Expr(:., m.finalacc, Expr(:quote, :val))), $gsym) ) )

		# enclose in a try block
		body = Expr(:try, Expr(:block, body...),
				          :e, 
				          quote 
				          	if isa(e, OutOfSupportError)
				          		return(-Inf, zero($PARAM_SYM))
				          	else
				          		throw(e)
				          	end
				          end)

	else  # case without gradient
		body = [ betaAssign(m)...,         # assigments beta vector -> model parameter vars
		         :($ACC_SYM = LLAcc(0.)),  # initialize accumulator
                 m.source.args...,         # model statements
                 :(return $( Expr(:., ACC_SYM, Expr(:quote, :val)) )) ]

		# enclose in a try block
		body = Expr(:try, Expr(:block, body...),
				          :e, 
				          :(if isa(e, OutOfSupportError); return(-Inf); else; throw(e); end) )

		header = Expr[]
	end

	# identify external vars and add definitions x = Main.x
	ev = setdiff(m.accanc, union(m.varsset, Set(ACC_SYM), Set(collect(keys(m.pars))...))) # vars that are external to the model
	header = [[ :( local $v = $(Expr(:., :Main, Expr(:quote, v))) ) for v in ev]..., header...] # assigment block

	# build and evaluate the let block containing the function and external vars hooks
	fn = newvar(:ll)
	body = Expr(:function, Expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	Expr(:block, body) )
	body = Expr(:let, Expr(:block, :(global $fn), header..., body))

	# println("#############\n$body\n############")

	debug ? body : (eval(body) ; (eval(fn), m.bsize, m.pars, m.init) )
end


######### rewrites model expression to make it differentiable  ###########
#  - rewrite ~ operators  as acc += logpdf..(=)
function parseModel!(m::ParsingStruct, source::Expr)
	local distribFound::Bool = false

	explore(ex::Expr)       = explore(toExH(ex))
	explore(ex::ExH)      = error("[parseModel] unmanaged expr type $(ex.head)")
	explore(ex::ExLine)   = nothing  # remove #line statements
	explore(ex::ExRef)    = toExpr(ex) # no processing
	explore(ex::ExEqual)  = toExpr(ex) # no processing
	explore(ex::ExVcat)   = toExpr(ex) # no processing
	explore(ex::ExDot)    = toExpr(ex) # no processing
	explore(ex::ExPEqual) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :+, args...)) )
	explore(ex::ExMEqual) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :-, args...)) )
	explore(ex::ExTEqual) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :*, args...)) )
	explore(ex::ExTrans)  = explore(Expr(:call, :transpose, ex.args[1]))

	function explore(ex::ExBlock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(ex2)
				ex3==nothing ? nothing : push!(al, ex3)
			else
			push!(al, ex2)
			end
		end
		Expr(ex.head, al...)
	end

	function explore(ex::ExCall)
		ex.args[1] != :~ && return toExpr(ex)
		distribFound = true
		fn = symbol("logpdf$(ex.args[3].args[1])")
		return :($ACC_SYM = $ACC_SYM + logpdf( $(ex.args[3]) , $(ex.args[2])) )
	end

	# blockify if single expression
	source.head == :block || ( source = Expr(:block, source) )

	assert(length(source.args)>=1, "model should have at least 1 statement")

	m.source = explore(source)

	# if no distribution expression '~' was found consider that last expr is the variable to be maximized 
	if !distribFound
		lastex = m.source.args[end]
		lastex.head == :(=) && (lastex = lastex.args[2]) # if assigment, take rhs only 
		m.source.args[end] = :( $ACC_SYM = $ACC_SYM + $lastex ) 
	end
end
