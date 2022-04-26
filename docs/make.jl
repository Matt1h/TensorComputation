using TensorToolkit
using Documenter

DocMeta.setdocmeta!(TensorToolkit, :DocTestSetup, :(using TensorToolkit); recursive=true)

makedocs(;
    modules=[TensorToolkit],
    authors="Matthias Holzenkamp",
    repo="https://github.com/Matt1h/TensorToolkit.jl/blob/{commit}{path}#{line}",
    sitename="TensorToolkit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Matt1h.github.io/TensorToolkit.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Matt1h/TensorToolkit.jl",
    devbranch="master",
)
