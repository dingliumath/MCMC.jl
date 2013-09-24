
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
				error("[generateModelFunction] invalid gradient var type $v $(typeof(vh))")
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
