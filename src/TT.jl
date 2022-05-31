using SparseArrays
using LinearAlgebra
using TensorOperations
using StaticArrays


mutable struct TTCore{T, N} <: AbstractArray{T, N}
    a::Array{T, N}
    function TTCore(a::Array{T, 3}) where {T}
        return new{T, 3}(a)
    end
end

@inline Base.@propagate_inbounds Base.size(C::TTCore) = Tuple(size(C.a))
@inline Base.@propagate_inbounds Base.getindex(C::TTCore, i::Int) = getindex(C.a, i)
@inline Base.@propagate_inbounds Base.getindex(C::TTCore, I::Vararg{Int, 2}) = getindex(C.a, I...)


struct TT{T<:Real, N, T1<:AbstractArray, T2<:AbstractString, T3<:Int, T4<:AbstractArray} <: AbstractArray{T, N}
    cores::T1
    DeCompType::T2
    order::T3
    ranks::T4
    ColDims::T4
    RowDims::T4

    function TT(X::AbstractArray{T, N}, psi::AbstractArray, TT_type::AbstractString, add_one::Bool=true) where{T, N}
        d, m = size(X)  
        p = length(psi)
        order = p + 1

        # dims
        ranks = Vector{Int}(undef, order+1)
        ranks[1] = 1
        ranks[order+1] = 1
        [ranks[i] = m for i in range(2, order)]
        
        ColDims = repeat([1], order)
        RowDims = push!(repeat([d + add_one], order-1), m)

        # cores
        cores = map(collect(range(1, order))) do i
            if i == 1  # first core
                core = zeros(1, d + add_one, m)
                if add_one
                    core[1, 1, :] .= 1
                    core[1, 2:end, :] = psi[i].(X)
                else
                    core[1, :, :] = psi[i].(X)
                end
                return TTCore(core)
            elseif i == order # last core
                core = zeros(m, m, 1)
                for i in range(1, m)
                    core[i, i, 1] = 1
                end
                return TTCore(core)
            else  # middel cores
                core = zeros(m, d + add_one, m)
                if add_one
                    for j in range(1, m)
                        core[j, 1, j] = 1
                        core[j, 2:end, j] = psi[i].(X)[:, j]
                    end
                else
                    for j in range(1, m)
                        core[j, :, j] = psi[i].(X)[:, j]
                    end
                end
                return TTCore(core)
            end
        end

        return new{T, N, typeof(cores), typeof(TT_type), typeof(order),
         typeof(ranks)}(cores, TT_type, order, ranks, ColDims, RowDims)
    end

    function TT(M::AbstractArray{T, N}, eps=1e-14) where {T, N}  
        RowDims = collect(size(M))    
        order = length(RowDims)
        ranks = ones(Int, order+1)
        ColDims = repeat([1], order)
        TT_type = "from_full_Array"    
        
        # cores
        cores = []

        ep = eps/sqrt(order-1)
        for i = 1:order-1
            m = RowDims[i]*ranks[i]
            M = reshape(M, m, Int(length(M)/m))
            u, s, v = svd(M)
            r1 = chop(s, ep*norm(s))
            u = u[:, 1:r1]
            s = s[1:r1]
            ranks[i+1] = r1
            push!(cores, TTCore(reshape(u, ranks[i], RowDims[i], ranks[i+1])))
            v = v[:, 1:r1]
            M = Matrix((v*diagm(s))')
        end

        # last core
        push!(cores, TTCore(reshape(M, ranks[order], RowDims[order], ranks[order+1])))

        return new{T, N, typeof(cores), typeof(TT_type), typeof(order), 
        typeof(ranks)}(cores, TT_type, order, ranks, ColDims, RowDims)
    end
end


@inline Base.@propagate_inbounds Base.size(A::TT) = Tuple(A.RowDims)
@inline Base.@propagate_inbounds Base.getindex(A::TT, i::Int) = getindex(matricize(A), i)
@inline Base.@propagate_inbounds Base.getindex(A::TT, I::Vararg{Int, 2}) = getindex(matricize(A), I...)


function chop(M::AbstractArray, eps=1e-14::Real)
    # Check for zero tensor
    if norm(M) == 0
        return 1
    end

    # Check for zero tolerance
    if  eps <= 0
        return length(M)
    end
    
    M0 = cumsum(M[end:-1:1].^2)
    ff = findall(M0 .< eps.^2)

    if isempty(ff)
        return length(M0)
    else
        return length(M0) - ff[end]
    end
end


function ortho_left!(A::TT, start_idx::Int, end_idx::Int)
    for i in range(start_idx, end_idx)
        u, s, v = svd(left_unfolding(A.cores[i]))
        A.ranks[i+1] = size(u, 2)
        A.cores[i] = left_folding(u, A.ranks[i], A.RowDims[i])
        W = diagm(s)*v'
        @tensor A.cores[i+1].a[a, b, c] := W[a, m]*A.cores[i+1].a[m, b, c]
    end
    return A
end


function ortho_right!(A::TT, start_idx::Int, end_idx::Int)
    for i in range(start=start_idx, stop=end_idx, step=-1)
        u, s, v = svd(right_unfolding(A.cores[i]))
        A.ranks[i] = size(v', 1)
        A.cores[i] = right_folding(v', A.RowDims[i], A.ranks[i+1])
        W = left_unfolding(A.cores[i - 1])*u*diagm(s)
        A.cores[i-1] = left_folding(W, A.ranks[i-1], A.RowDims[i-1])
    end
    return A
end


function pinv(A::TT, idx::Int)
    W = deepcopy(A)
    ortho_left!(W, 1, idx-1)
    ortho_right!(W, idx+1, W.order)
    u, s, v = svd(left_unfolding(W.cores[idx]))
    W.ranks[idx+1] = size(u, 2)

    # update (idx)th core
    W.cores[idx] = left_folding(u, W.ranks[idx], W.RowDims[idx])

    # update (idx+1)th core
    @tensor W.cores[idx+1].a[a, b, c] := v'[a, m]*W.cores[idx+1].a[m, b, c]

    # TODO: code to this line: separate TT svd function 

    # contract s with (idx+1)th core
    @tensor W.cores[idx+1].a[a, b, c] := diagm(1 ./ s)[a, m]*W.cores[idx+1].a[m, b, c]
    return W
end


function left_unfolding(core::TTCore)
    r1 = size(core, 1)
    n = size(core, 2)
    r2 = size(core, 3)
    return reshape(core.a, r1*n, r2)
end


function left_folding(A::AbstractArray, r1::Int, n::Int)
    r2 = size(A, 2)
    return TTCore(reshape(A, r1, n, r2))
end


function right_unfolding(core::TTCore)
    r1 = size(core, 1)
    n = size(core, 2)
    r2 = size(core, 3)
    return reshape(core.a, r1, n*r2)
end


function right_folding(A::AbstractArray, n::Int, r2::Int)
    r1 = size(A, 1)
    return TTCore(reshape(A, r1, n, r2))
end


function full(A::TT)
    M = A.cores[1].a
    for q in range(1, A.order-1)
        B = reshape(M, (prod(A.RowDims[1:q]), A.ranks[q+1]))
        @tensor M[i, j, k] := B[i, l]*A.cores[q+1].a[l, j, k]
    end
    return reshape(M, size(A))
end


function matricize(A::TT{T}) where{T}
    M = A.cores[1].a
    for q in range(1, A.order-1)
        B = reshape(M, (prod(A.RowDims[1:q]), A.ranks[q+1]))
        @tensor M[i, j, k] := B[i, l]*A.cores[q+1].a[l, j, k]
    end
    return reshape(M, size(M, 1), size(M, 2))
end



# function TT_mul(A::TT, B::TT)
#     M = kron(A.cores[1], B.cores[1])
#     for i in range(1, A.order)
#         @tensor M[a, b] := M[kron(A.cores[i][:, a, :], B.cores[i][:, b, :])
#     end
#     return M
# end

psi = [sin, cos]
X = [3 4 1; 
    1 1.2 2; 
    1 2 3; 
    0 9 2]
X = X'

X_A = ones(Float64, 2, 3, 2)
X_A[1, 1, 1] = 1.7
X_A[2, 1, 2] = 6.8
X_A[2, 2, 2] = 2.4

M_A =  TT(X_A);
M = TT(X, psi, "function_major")
