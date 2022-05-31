module TensorComputation
abstract type AbstractTensor end

include("./TT.jl")
export TT

export left_unfolding
export left_folding
export right_unfolding
export right_folding

export ortho_left!
export ortho_right!

export pinv

export full
export matricize
end
