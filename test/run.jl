# using TensorComputation

# psi = [sin, cos]
# X = [3 4 1; 
#     1 1.2 2; 
#     1 2 3; 
#     0 9 2]
# X = X'

# M = TT(X, psi, "function_major")
# P = pinv(M, M.order-1)
# MM = matricize(M)


# struct Foo{T, N, T1} <: AbstractArray{T, N}
#     a::T1
#     function Foo(data::AbstractArray{T, N}) where {T <: Real, N}
#         return new{T, N, typeof(data)}(data)     
#     end
# end

# function size(A::Foo)
#     size(A.a)
# end

# struct StringIndexVector{T} <: AbstractVector{T}
#     svec::Vector{String}
#     index::Vector{Integer}
# end


struct FooArray{T, N, AT, VT} <: AbstractArray{T, N}
    data::AT
    vec::VT
    function FooArray(data::AbstractArray{T1, N}, vec::AbstractVector{T2}) where {T1 <: AbstractFloat, T2 <: AbstractFloat, N}
        # length(vec) == size(data, 2) || error("Inconsistent dimensions")
        new{T1, N, typeof(data), typeof(vec)}(data, vec)
    end
end
