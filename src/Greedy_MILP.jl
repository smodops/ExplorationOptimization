###############################################################################
#  Greedy_MILP.jl — Exploration-Optimization MILP
#
#  Solves the Exploration-Optimization MILP for the ExpOpt policy. The model
#  assigns K assortment "copies" per profile so that every product that needs
#  exploration (ExpReq[i,j] == 1) is covered by at least one copy.
###############################################################################

"""
    Greedy_MILP(vhat, inst, p, ExpReq, params) -> Array{Int,3}

Solves the Exploration-Optimization MILP and returns the exploration assortments for
every profile.

Arguments
---------
- `vhat`   : Estimated attractiveness matrix (N × M).
- `inst`   : `InstanceData` struct (N, M, C, w).
- `p`      : Arrival probability matrix (1 × M).
- `ExpReq`    : Binary coverage requirement matrix (N × M) obtained from the Greedy-heuristic; 
                1 means product i must be explored for profile j.
- `params` : `SimParams` struct (for `MIP_time_limit`).

Returns
-------
- `zopt` : Binary assortment array of shape (N × M × K).
"""
function Greedy_MILP(vhat::Matrix{Float64}, inst::InstanceData,
                     ExpReq::Matrix{Float64}, params::SimParams)::Array{Int,3}
    N = inst.N;  M = inst.M;  C = inst.C;  w = inst.w; p = inst.p;
    K = Int(ceil(N / C))   # maximum copies needed to cover all products

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_time_limit_sec(model, params.MIP_time_limit)
    set_optimizer_attribute(model, "MIPGap", params.MIP_gap)

    @variable(model, z[1:N, 1:M, 1:K], Bin)
    @variable(model, 0 <= y[1:N+1, 1:M, 1:K] <= 1)

    @objective(model, Max,
        sum(p[j] * w[i] * y[i, j, k] for i in 1:N, j in 1:M, k in 1:K))

    ## Alternative objective function (without arrival probabilities)    
    # @objective(model, Max,
    #     sum(w[i] * y[i, j, k] for i in 1:N, j in 1:M, k in 1:K))    

    # MNL constraints
    for i in 1:N, j in 1:M, k in 1:K
        v = vhat[i, j]
        @constraint(model, y[i,j,k] <= z[i,j,k] * v)
        @constraint(model, y[i,j,k] <= y[N+1,j,k] * v)
        @constraint(model, y[i,j,k] >= v * (y[N+1,j,k] - (1 - z[i,j,k])))
    end

    # Probability normalization 
    for j in 1:M, k in 1:K
        @constraint(model, y[N+1,j,k] + sum(y[i,j,k] for i in 1:N) == 1)
    end

    # Assortment capacity constraints
    for j in 1:M, k in 1:K
        @constraint(model, sum(z[i,j,k] for i in 1:N) <= C)
    end

    # Coverage requirement: each flagged (product, profile) must appear at least once
    for i in 1:N, j in 1:M
        if ExpReq[i, j] == 1
            @constraint(model, sum(z[i,j,k] for k in 1:K) >= 1)
        end
    end

    optimize!(model)

    return round.(Int, value.(z))
end
