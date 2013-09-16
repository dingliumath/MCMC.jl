module Abcd

	using Distributions
	using Base.LinAlg.BLAS

	##########  creates a parameterized type to ease AST exploration  ############
	type ExprH{H}
		head::Symbol
		args::Vector
		typ::Any
	end
	toExprH(ex::Expr) = ExprH{ex.head}(ex.head, ex.args, ex.typ)
	toExpr(ex::ExprH) = Expr(ex.head, ex.args...)

	typealias Exprequal    ExprH{:(=)}
	typealias Exprdcolon   ExprH{:(::)}
	typealias Exprpequal   ExprH{:(+=)}
	typealias Exprmequal   ExprH{:(-=)}
	typealias Exprtequal   ExprH{:(*=)}
	typealias Exprtrans    ExprH{symbol("'")} 
	typealias Exprcall     ExprH{:call}
	typealias Exprblock	   ExprH{:block}
	typealias Exprline     ExprH{:line}
	typealias Exprvcat     ExprH{:vcat}
	typealias Exprref      ExprH{:ref}
	typealias Exprif       ExprH{:if}
	typealias Exprcomp     ExprH{:comparison}
	typealias Exprdot      ExprH{:.}

	## variable symbol survey functions
	getSymbols(ex::Any) =        Set{Symbol}()
	getSymbols(ex::Symbol) =     Set{Symbol}(ex)
	getSymbols(ex::Array) =      mapreduce(getSymbols, union, ex)
	getSymbols(ex::Expr) =       getSymbols(toExprH(ex))
	getSymbols(ex::ExprH) =      mapreduce(getSymbols, union, ex.args)
	getSymbols(ex::Exprcall) =   mapreduce(getSymbols, union, ex.args[2:end])  # skip function name
	getSymbols(ex::Exprref) =    setdiff(mapreduce(getSymbols, union, ex.args), Set(:(:), symbol("end")) )# ':'' and 'end' do not count
	getSymbols(ex::Exprcomp) =   setdiff(mapreduce(getSymbols, union, ex.args), 
		Set(:(>), :(<), :(>=), :(<=), :(.>), :(.<), :(.<=), :(.>=), :(==)) )

	getSymbols(ex::Exprdot) =     Set{Symbol}(ex.args[1])  # return variable, not fields

	## variable symbol subsitution functions
	substSymbols(ex::Any, smap::Dict) =           ex
	substSymbols(ex::Expr, smap::Dict) =          substSymbols(toExprH(ex), smap::Dict)
	substSymbols(ex::Vector{Expr}, smap::Dict) =  map(e -> substSymbols(e, smap), ex)
	substSymbols(ex::ExprH, smap::Dict) =         Expr(ex.head, map(e -> substSymbols(e, smap), ex.args)...)
	substSymbols(ex::Exprcall, smap::Dict) =      Expr(:call, ex.args[1], map(e -> substSymbols(e, smap), ex.args[2:end])...)
	substSymbols(ex::Exprdot, smap::Dict) =       (ex.args[1] = substSymbols(ex.args[1], smap) ; toExpr(ex) )
	substSymbols(ex::Symbol, smap::Dict) =        get(smap, ex, ex)

	
	# naming conventions + functions
	const ACC_SYM = :_acc       # name of accumulator variable
	const PARAM_SYM = :_beta    # name of parameter vector
	const TEMP_NAME = "tmp"     # prefix of temporary variables in log-likelihood function
	const DERIV_PREFIX = "_d"   # prefix of gradient variables

	dprefix(v::Union(Symbol, String, Char)) = symbol("$DERIV_PREFIX$v")
	dprefix(v::Expr) = dprefix(toExprH(v))
	dprefix(v::Exprref) = Expr(:ref, dprefix(v.args[1]), v.args[2:end]...)
	dprefix(v::Exprdot) = Expr(:., dprefix(v.args[1]), v.args[2:end]...)

	isSymbol(ex)   = isa(ex, Symbol)
	isDot(ex)      = isa(ex, Expr) && ex.head == :.   && isa(ex.args[1], Symbol)
	isRef(ex)      = isa(ex, Expr) && ex.head == :ref && isa(ex.args[1], Symbol)

	# var name generator
	let
		vcount = Dict()
		global newvar
		function newvar(radix::Union(String, Symbol)="")
			vcount[radix] = haskey(vcount, radix) ? vcount[radix]+1 : 1
			return symbol("$radix#$(vcount[radix])")
		end
	end

	immutable PDims
		pos::Integer   # starting position of parameter in the parameter vector
		dims::Tuple    # dimensions of user facing parameter, can be a scalar, vector or matrix
	end

	######### structure for parsing model  ##############
	type ParsingStruct
		bsize::Int                # length of beta, the parameter vector
		pars::Dict{Symbol, PDims} # parameters with their mapping to the beta real vector
		init::Vector{Float64}     # initial values of beta
		source::Expr              # model source, after first pass
		exprs::Vector{Expr}       # vector of assigments that make the model
		dexprs::Vector{Expr}      # vector of assigments that make the gradient
		finalacc::Symbol          # last symbol of loglik accumulator after renaming
		varsset::Set{Symbol}      # all the vars set in the model
		pardesc::Set{Symbol}      # all the vars set in the model that depend on model parameters
		accanc::Set{Symbol}       # all the vars (possibly external) that influence the accumulator
	end
	ParsingStruct() = ParsingStruct(0, Dict{Symbol, PDims}(), Float64[], :(), Expr[], Expr[], ACC_SYM, 
		Set{Symbol}(), Set{Symbol}(), Set{Symbol}())

	#### Log-likelihood accumulator type  ####
	# this makes the model function easier to generate compared to a Float64
	#   - embeds the error throwing when log-likelihood reaches -Inf
	#   - calculates the sum when logpdf() returns an Array
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


	######### first pass on the model
	#  - rewrite ~ operators  as acc += logpdf..(=)
	#  - translates x += y into x = x + y, same for -= and *=
	function parseModel!(m::ParsingStruct, source::Expr)
		local distribFound::Bool = false

		explore(ex::Expr) =       explore(toExprH(ex))
		explore(ex::ExprH) =      error("[parseModel] unmanaged expr type $(ex.head)")
		explore(ex::Exprline) =   nothing  # remove #line statements
		explore(ex::Exprref) =    toExpr(ex) # no processing
		explore(ex::Exprequal) =  toExpr(ex) # no processing
		explore(ex::Exprvcat) =   toExpr(ex) # no processing

		explore(ex::Exprdot) =    toExpr(ex) # no processing
		
		explore(ex::Exprpequal) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :+, args...)) )
		explore(ex::Exprmequal) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :-, args...)) )
		explore(ex::Exprtequal) = (args = ex.args ; Expr(:(=), args[1], Expr(:call, :*, args...)) )

		explore(ex::Exprtrans) = explore(Expr(:call, :transpose, ex.args[1]))

		function explore(ex::Exprblock)
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

		function explore(ex::Exprcall)
			ex.args[1] != :~ && return toExpr(ex)

			distribFound = true
			fn = symbol("logpdf$(ex.args[3].args[1])")
			# return :($ACC_SYM = $ACC_SYM + $(Expr(:call, fn, ex.args[3].args[2:end]..., ex.args[2])))
			return :($ACC_SYM = $ACC_SYM + logpdf( $(ex.args[3]) , $(ex.args[2])) )
		end

		# blockify if single expression
		source.head == :block || ( source = Expr(:block, source) )

		assert(length(source.args)>=1, "model should be a block with at least 1 statement")

		m.source = explore(source)

		# if no distribution expression '~' was found consider that last expr is the variable to be maximized 
		if !distribFound
			# m.source.args[end] = Expr(:(=) , ACC_SYM, m.source.args[end] )
			lastex = m.source.args[end]
			lastex.head == :(=) && (lastex = lastex.args[2]) # if assigment, take rhs only 
			m.source.args[end] = :( $ACC_SYM = $ACC_SYM + $lastex ) 
		end
	end

	######## unfolds expressions to prepare derivation ###################
	function unfold!(m::ParsingStruct)

		explore(ex::Expr) =       explore(toExprH(ex))
		explore(ex::ExprH) =      error("[unfold] unmanaged expr type $(ex.head) in ($ex)")
		explore(ex::Exprline) =   nothing     # remove line info
		explore(ex::Exprref) =    toExpr(ex)   # unchanged
		explore(ex::Exprcomp) =   toExpr(ex)  # unchanged
		explore(ex::Exprvcat) =   explore(Expr(:call, :vcat, ex.args...) )  # translate to vcat(), and explore
		explore(ex::Exprtrans) =  explore(Expr(:call, :transpose, ex.args[1]) )  # translate to transpose() and explore
		explore(ex::Any) =        ex

		explore(ex::Exprdot) =    toExpr(ex)   # unchanged

		explore(ex::Exprblock) =  mapreduce(explore, (a,b)->b, ex.args)  # process, and return last evaluated
		
		function explore(ex::Exprequal) 
			lhs = ex.args[1]
			assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
				"[unfold] not a symbol on LHS of assigment $ex")

			rhs = ex.args[2]
			if isSymbol(rhs) || isa(rhs, Real) || isDot(rhs)
				push!(m.exprs, Expr(:(=), lhs, rhs))
			elseif isa(rhs, Expr) 
					ue = explore(toExprH(rhs)) # explore will return something in this case
					push!(m.exprs, Expr(:(=), lhs, ue))
			else  # unmanaged kind of rhs
			 	error("[unfold] can't handle RHS of assignment $(toExpr(ex))")
			end
			lhs
		end

		function explore(ex::Exprcall) 
			na = {ex.args[1]}   # function name
			args = ex.args[2:end]  # arguments

			# if more than 2 arguments, +, sum and * are converted  to nested expressions
			#  (easier for derivation)
			#  TODO : apply to max, min
			# TODO : apply to other n-ary (n>2) operators ?
			if contains([:+, :*, :sum], na[1]) 
				while length(args) > 2
					a2 = pop!(args)
					a1 = pop!(args)
					push!(args, Expr(:call, ex.args[1], a1, a2))
				end
			end

			for e2 in args  
				if isa(e2, Expr) # only refs and calls will work
					ue = explore(e2)
					nv = newvar(TEMP_NAME)
					push!(m.exprs, :($nv = $ue))
					push!(na, nv)
				else
					push!(na, e2)
				end
			end

			Expr(ex.head, na...)
		end

		explore(m.source)
	end

	######### renames variables set several times to make them unique  #############
	# FIXME : algo doesn't work when a variable sets individual elements, x = .. then x[3] = ...; 
	# FIXME 2 : external variables redefined within model are not renamed
	function uniqueVars!(m::ParsingStruct)
		el = m.exprs
	    subst = Dict{Symbol, Symbol}()
	    used = Set(ACC_SYM)

	    for idx in 1:length(el) # idx=4
	        # first, substitute in the rhs the variables names that have been renamed
	        el[idx].args[2] = substSymbols(el[idx].args[2], subst)

	        # second, rename lhs symbol if set before
	        lhs = collect(getSymbols(el[idx].args[1]))[1]  # there should be only one
	        if contains(used, lhs) # if var already set once => create a new one
	            subst[lhs] = newvar(lhs) # generate new name, add it to substitution list for following statements
	            el[idx].args[1] = substSymbols(el[idx].args[1], subst)
	        else # var set for the first time
	            union!(used, Set(lhs)) 
	        end
	    end

		m.finalacc = haskey(subst, ACC_SYM) ? subst[ACC_SYM] : ACC_SYM  # keep reference of potentially renamed accumulator
	end

	######### identifies vars #############
	# - lists variables that depend on model parameters 
	# - lists variables that influence the accumulator
	# - lists variables defined
	# In order to 
	#   1) restrict gradient code to the strictly necessary variables 
	#   2) move parameter independant variables definition out the function (but within closure) 
	#   3) TODO : remove unnecessary variables (with warning)
	#   4) identify external vars
	function categorizeVars!(m::ParsingStruct) 
		lhsSymbol(ex) = Set(isSymbol(ex.args[1]) ? ex.args[1] : ex.args[1].args[1])

	    m.varsset = mapreduce(lhsSymbol, union, m.exprs)

	    local parset = Set{Symbol}(collect(keys(m.pars))...)  # [p.sym for p in m.pars]...)
	    m.pardesc = copy(parset)  # start with parameter symbols
	    for ex2 in m.exprs 
	        lhs = lhsSymbol(ex2)
	        rhs = getSymbols(ex2.args[2])

	        !isempty(intersect(rhs, m.pardesc)) && union!(m.pardesc, lhs)
	    end

	    m.accanc = Set{Symbol}(m.finalacc)
	    for ex2 in reverse(m.exprs) # proceed backwards ex2 = reverse(m.exprs)[3]
	        lhs = lhsSymbol(ex2)
	        rhs = setdiff(getSymbols(ex2), lhs) # to pickup potential index on lhs as an ancestor

	        !isempty(intersect(lhs, m.accanc)) && union!(m.accanc, rhs)
	    end

	    contains(m.pardesc, m.finalacc) || warn("Model parameters do not seem to influence model outcome")

	    local parset2 = setdiff(parset, m.accanc)
	    isempty(parset2) || warn("Model parameter(s) $(collect(parset2)) do not seem to influence model outcome")

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

	######## sets inital values from 'init' given as parameter  ##########
	function setInit!(m::ParsingStruct, init)
	    assert(length(init)>=1, "There should be at leat one parameter specified, none found")

	    for p in init  # p = collect(init)[1]
	        par = p[1]  # param symbol defined here
	        def = p[2]

	        assert(typeof(par) == Symbol, "[setInit] not a symbol in init param : $(par)")

	        if isa(def, Real)  #  single param declaration
	            # push!(m.pars, MCMCParams(par, Integer[], m.bsize+1)) 
	            m.pars[par] = PDims(m.bsize+1, ())
	            m.bsize += 1
	            push!(m.init, def)

	        elseif isa(def, Array) && ndims(def) == 1
	            nb = size(def,1)
	            # push!(m.pars, MCMCParams(par, Integer[nb], (m.bsize+1):(m.bsize+nb)))
	            m.pars[par] = PDims(m.bsize+1, (nb,))
	            m.bsize += nb
	            m.init = [m.init, def...]

	        elseif isa(def, Array) && ndims(def) == 2
	            nb1, nb2 = size(def)
	            # push!(m.pars, MCMCParams(par, Integer[nb1, nb2], (m.bsize+1):(m.bsize+nb1*nb2))) 
	            m.pars[par] = PDims(m.bsize+1, (nb1,nb2))
	            m.bsize += nb1*nb2
	            m.init = [m.init, vec(def)...]

	        else
	            error("[setInit] unsupported parameter type for $(par)")
	        end
	    end
	end

	function betaAssign(m::ParsingStruct)
		assigns = Expr[]
		for p in keys(m.pars)
			v = m.pars[p]

			if length(v.dims) == 0  # scalar
				push!(assigns, :($p = $PARAM_SYM[ $(v.pos) ]) )

			elseif length(v.dims) == 1  # vector
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(assigns, :($p = $PARAM_SYM[ $(Expr(:quote,r)) ]) )

			else # matrix case  (needs a reshape)
				r = v.pos:(v.pos+prod(v.dims)-1)
				push!(assigns, :($p = reshape($PARAM_SYM[ $(Expr(:quote,r)) ], $(v.dims[1]), $(v.dims[2]))) )
			end
		end			
		assigns
	end

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
		
		# println(body)

		eval(body)
		fn = eval(fn)

		# now evaluate vhint (or throw error if model does not evaluate for given initial values)
		res = fn(m.init)
		!isa(res, Real) && error("Model outcome should be a scalar, $(typeof(res)) found")
		res == -Inf && error("Initial values out of model support, try other values")
	end


	######### builds the model function ##############
	# 'init' contains the dictionary of model params and their initial value
	#    ! Initial values are used for the precalculate run that will allow 
	#    to know all variables types.
	# If 'debug' is set to true, the function prints out the model function 
	#  that would have been created
	#
	function generateModelFunction(model::Expr; gradient=false, debug=false, init...)
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
					push!(body, :( $dsym = zeros(Float64, $(Expr(:tuple,size(vh)...)))) )
				elseif 	isa(vh, Distribution)  #  TODO : find real equivalent vector size
					push!(body, :( $(symbol("$dsym#1")) = 0. ) )
					push!(body, :( $(symbol("$dsym#2")) = 0. ) )
				elseif 	isa(vh, Array) && isa(vh[1], Distribution)  #  TODO : find real equivalent vector size
					push!(body, :( $(symbol("$dsym#1")) = zeros(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
					push!(body, :( $(symbol("$dsym#2")) = zeros(Float64, $(Expr(:tuple,size(vh)...)) ) ) )
					# push!(body, :( $dsym = zeros(Float64, $(Expr(:tuple,size(vh)...,2))) ) )
				else
					error("[generateModelFunction] invalid gradient var type $v $(typeof(vh))")
				end
			end

			# build function statements, and move to let block constant statements for optimization
			header = Expr[]  # let block statements
			fvars = union(Set([e.args[1] for e in body]...), Set(PARAM_SYM)) # vars that are re-evaluated at each function call
			for ex in [m.exprs..., m.dexprs...]
				# if length(intersect(getSymbols(ex.args[2]), fvars)) > 0
					push!(body, ex)
					fvars = union(fvars, getSymbols(ex.args[1]))
				# else
				# 	push!(header, ex)
				# end
			end

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


	function debug(model::Expr; init...)
		m = ParsingStruct()
		setInit!(m, init)
		parseModel!(m, model)
		unfold!(m)
		uniqueVars!(m)
		categorizeVars!(m)

		println(Expr(:block, m.exprs...))
		
		preCalculate(m)
		backwardSweep!(m)

		println()
		println(Expr(:block, m.dexprs...))

	end


	include("mydiff.jl")
end 


