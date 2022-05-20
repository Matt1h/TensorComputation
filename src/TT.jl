using SparseArrays
using LinearAlgebra
using TensorOperations


struct TT{T1<:Real} <: AbstractTensor
    core::AbstractArray  # TODO: cores with own type, that includes DeCompType?
    DeCompType::AbstractString
    order::Int
    ranks::AbstractArray
    ColDim::AbstractArray
    RowDim::AbstractArray


    function TT(X::AbstractArray{T1}, psi::AbstractArray, TT_type::AbstractString, add_one::Bool=true) where{T1}
        d, m = size(X)  
        p = length(psi)
        order = p + 1
        cores = Array{AbstractArray}(undef, (p+1, 1))
        
        # first core
        cores[1] = zeros(1, d + add_one, m)
        if add_one
            cores[1][1, 1, :] .= 1
            cores[1][1, 2:end, :] = psi[1].(X)
        else
            cores[1][1, :, :] = psi[1].(X)
        end

        # middel cores
        for i in range(2, p)
            cores[i] = zeros(m, d + add_one, m)
            if add_one
                for j in range(1, m)
                    cores[i][j, 1, j] = 1
                    cores[i][j, 2:end, j] = psi[i].(X)[:, j]
                end
            else
                for j in range(1, m)
                    cores[i][j, :, j] = psi[i].(X)[:, j]
                end
            end
        end
    
        # last core
        cores[p+1] = zeros(m, m, 1)
        for i in range(1, m)
            cores[p+1][i, i, 1] = 1
        end
        
        # dims
        ranks = Array{Int}(undef, order+1, 1)
        for i in range(1, order)
            ranks[i] = size(cores[i], 1)
        end
        ranks[order+1] = size(cores[p+1], 3)
        
        ColDim = repeat([1], order)  # TODO: when isnt this 1?
        RowDim = push!(repeat([d + add_one], order-1), m)

        return new{T1}(cores, TT_type, order, ranks, ColDim, RowDim)
    end
end


function ortho_left!(A::TT, start_idx::Int, end_idx::Int)
    for i in range(start_idx, end_idx)
        u, s, v = svd(left_unfolding(A.core[i]))
        A.ranks[i+1] = size(u, 2)
        A.core[i] = left_folding(u, A.ranks[i], A.RowDim[i])
        W = diagm(s)*v'
        @tensor A.core[i+1][a, b, c] := W[a, m]*A.core[i+1][m, b, c]
    end
    return A
end


function ortho_right!(A::TT, start_idx::Int, end_idx::Int)
    for i in range(start=start_idx, stop=end_idx, step=-1)
        u, s, v = svd(right_unfolding(A.core[i]))
        A.ranks[i] = size(v', 1)
        A.core[i] = right_folding(v', A.RowDim[i], A.ranks[i+1])
        W = left_unfolding(A.core[i - 1])*u*diagm(s)
        A.core[i-1] = left_folding(W, A.ranks[i-1], A.RowDim[i-1])
    end
    return A
end


function pinv(A::TT, idx::Int)
    W = deepcopy(A)
    ortho_left!(W, 1, idx-1)
    ortho_right!(W, idx+1, W.order)
    u, s, v = svd(left_unfolding(W.core[idx]))
    W.ranks[idx+1] = size(u, 2)

    # update (idx)th core
    W.core[idx] = left_folding(u, W.ranks[idx], W.RowDim[idx])

    # update (idx+1)th core
    @tensor W.core[idx+1][a, b, c] := v'[a, m]*W.core[idx+1][m, b, c]

    # TODO: code to this line: separate TT svd function 

    # contract s with (idx+1)th core
    @tensor W.core[idx+1][a, b, c] := diagm(1 ./ s)[a, m]*W.core[idx+1][m, b, c]
    return W
end


function left_unfolding(core::AbstractArray)  # TODO: cores with own type?
    r1 = size(core, 1)
    n = size(core, 2)
    r2 = size(core, 3)
    return reshape(core, r1*n, r2)
end


function left_folding(A::AbstractArray, r1::Int, n::Int)
    r2 = size(A, 2)
    return reshape(A, r1, n, r2)
end


function right_unfolding(core::AbstractArray)  # TODO: cores with own type?
    r1 = size(core, 1)
    n = size(core, 2)
    r2 = size(core, 3)
    return reshape(core, r1, n*r2)
end


function right_folding(A::AbstractArray, n::Int, r2::Int)
    r1 = size(A, 1)
    return reshape(A, r1, n, r2)
end


function matricize(A::TT{T}) where{T}
    M = A.core[1]
    for q in range(1, A.order-1)
        B = reshape(M, (prod(A.RowDim[1:q]), A.ranks[q+1]))
        @tensor M[i, j, k] := B[i, l]*A.core[q+1][l, j, k]
    end
    return reshape(M, size(M, 1), size(M, 2))
end
