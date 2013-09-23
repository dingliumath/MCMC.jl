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
		# TODO : apply to max, min
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
