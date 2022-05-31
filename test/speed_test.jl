using TensorComputation
using Random
using TensorOperations
using Tullio
using LoopVectorization
using BenchmarkTools



function SpeedTest1()
    m = 400
    n = 300
    o = 280
    A = rand(m, n, o)
    W = rand(m, m)
    @tensor A[a, b, c] := W[a, l]*A[l, b, c]
end

function SpeedTest2()
    m = 400
    n = 300
    o = 280
    A = rand(m, n, o)
    W = rand(m, m)
    @tullio A[a, b, c] = W[a, l]*A[l, b, c]
end

@benchmark SpeedTest2()