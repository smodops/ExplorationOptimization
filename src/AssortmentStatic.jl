###############################################################################
#  AssortmentStatic.jl — Optimal static assortment via MILP
#
#  Solves the MNL assortment optimization problem for a single customer profile
#  using a MILP formulation.
###############################################################################

"""
    AssortmentStatic(v, inst) -> (Float64, Vector{Int})

Compute the revenue-maximizing assortment for a single customer profile.

The problem is solved as a mixed-integer linear program (MILP) using the
standard linearisation of the MNL choice probability:

    max  Σᵢ wᵢ · yᵢ
    s.t. yᵢ ≤ zᵢ · vᵢ          ∀i
         yᵢ ≤ y₀ · vᵢ           ∀i
         yᵢ ≥ vᵢ · (y₀ − 1 + zᵢ) ∀i
         y₀ + Σᵢ yᵢ = 1
         Σᵢ zᵢ ≤ C
         zᵢ ∈ {0,1}

Arguments
---------
- `v`    :  Product exponentiated mean utility (attractiveness) vector for the profile (length N).
- `inst` : `InstanceData` struct carrying N, C, and w.

Returns
-------
- Optimal expected revenue (Float64).
- Optimal assortment indicator vector (Vector{Int}, length N).
"""
function AssortmentStatic(v::Vector{Float64}, inst::InstanceData)
    N = inst.N;  C = inst.C;  w = inst.w

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(model, z[1:N], Bin)
    @variable(model, 0 <= y[1:N+1] <= 1)

    @objective(model, Max, sum(w[i] * y[i] for i in 1:N))

    for i in 1:N
        @constraint(model, y[i] <= z[i] * v[i])
        @constraint(model, y[i] <= y[N+1] * v[i])
        @constraint(model, y[i] >= v[i] * (y[N+1] - (1 - z[i])))
    end

    @constraint(model, sum(y[i] for i in 1:N+1) == 1)
    @constraint(model, sum(z[i] for i in 1:N) <= C)

    optimize!(model)

    obj  = objective_value(model)
    zopt = floor.(Int, round.(value.(z)))   # round to avoid numerical noise

    return obj, zopt
end
