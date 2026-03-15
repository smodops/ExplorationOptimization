###############################################################################
#  RLB_Solvers.jl — LP/MILP solvers for the RLB policy
#
#  Contains four inter-related functions:
#    ColGen        — column generation loop
#    RLB_Dual      — Dual LP of the RLB problem
#    SubProblem    — column-generation pricing subproblem (MILP)
#    RLB_Primal    — Primal LP of the RLB problem
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  Column generation loop
# ─────────────────────────────────────────────────────────────────────────────

"""
    ColGen(inst, Nhat, Di, Ki, E1, E2, vhat, f) -> (Matrix{Int}, Vector{Float64})

Run the column generation algorithm to solve the RLB dual to optimality.

Starting from the initial column set (E1, E2), iteratively:
1. Solve the dual LP.
2. For each profile, solve the pricing subproblem.
3. Add columns with positive reduced cost.
4. Terminate when no improving column exists.

Returns
-------
- `E1` : Updated assortment-profile matrix (N+1 × Esize).
- `E2` : Updated optimality gap vector (length Esize).
"""
function ColGen(inst::InstanceData,
                Nhat::Matrix{Int}, Di::Matrix{Int}, Ki::Matrix{Float64},
                E1::Matrix{Int}, E2::Vector{Float64},
                vhat::Matrix{Float64}, f::Matrix{Float64})
    N = inst.N;  M = inst.M;  C = inst.C;  w = inst.w

    tolerance = 10^-6

    # Build full E matrix (N+2 rows) from the split E1/E2
    E_full = vcat(E1, E2')

    opt_gap_now = 0.0 # Track the maximum optimality gap to avoid numerical issues in the stopping criterion

    while true
        _, _, dual_yopt = RLB_Dual(inst, Nhat, Di, Ki, E1, E2)

        cols_added = 0
        
        opt_gap_prior = opt_gap_now
        opt_gap_now = -1000.0

        for m in 1:M
            sub_obj, sub_z, _ = SubProblem(inst, vhat[:, m], dual_yopt, m)

            opt_gap_now = max(sub_obj - f[m], opt_gap_now)

            sub_obj - f[m] <= tolerance && continue   # no improving column (reduced cost ≤ 0)

            # Compute optimality gap for the new column
            v_s   = sub_z .* vhat[:, m]
            rev   = dot(w[1:N], v_s ./ (1.0 + sum(v_s)))
            delta = max(f[m] - rev, 0.0)

            new_col  = [sub_z; m; delta]
            E_full   = hcat(E_full, new_col)
            cols_added += 1
        end

        if cols_added == 0 || (abs(opt_gap_now - opt_gap_prior) <= tolerance)
            break   # optimality confirmed (stopping criterion includes a check for numerical stability)
        end

        # Re-split E into E1 / E2
        E1 = Matrix{Int}(round.(E_full[1:N+1, :], digits=0))
        E2 = Vector{Float64}(E_full[N+2, :])
    end

    return E1, E2
end

# ─────────────────────────────────────────────────────────────────────────────
#  Dual LP
# ─────────────────────────────────────────────────────────────────────────────

"""
    RLB_Dual(inst, Nhat, Di, Ki, E1, E2) -> (Float64, Vector, Matrix)

Solve the RLB dual LP.

Returns
-------
- `obj`   : Optimal dual objective value.
- `xopt`  : Optimal x variables (length N).
- `yopt`  : Optimal y variables (N × M).
"""
function RLB_Dual(inst::InstanceData,
                  Nhat::Matrix{Int}, Di::Matrix{Int},
                  Ki::Matrix{Float64},
                  E1::Matrix{Int}, E2::Vector{Float64})
    N = inst.N;  M = inst.M

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(model, x[1:N] >= 0)
    @variable(model, y[1:N, 1:M] >= 0)

    @objective(model, Max, sum(Ki[i,1] * x[i] for i in 1:N if Nhat[i,1] == 1))

    for i in 1:N
        Nhat[i, 1] == 0 && continue
        for j in 1:M
            if Di[i,j]==1    
                @constraint(model, x[i] <= y[i,j])
            end
        end
    end

    for e in 1:size(E1, 2)
        m      = E1[N+1, e]
        Δ      = E2[e]
        @constraint(model, sum(y[i, m] * E1[i, e] for i in 1:N) <= Δ)
    end

    optimize!(model)
    return objective_value(model), value.(x), value.(y)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Column-generation pricing subproblem
# ─────────────────────────────────────────────────────────────────────────────

"""
    SubProblem(inst, v, dual_yopt, profile) -> (Float64, Vector{Int}, Vector)

Solve the column-generation pricing MILP for a given customer profile.

Finds the assortment that maximizes the reduced cost with respect to the
current dual variables `dual_yopt`.

Returns
-------
- `obj`   : Optimal objective value.
- `zopt`  : Optimal assortment indicator (Vector{Int}, length N).
- `fopt`  : Optimal purchase-probability variables (Vector, length N+1).
"""
function SubProblem(inst::InstanceData, v::Vector{Float64},
                    dual_yopt::Matrix{Float64}, profile::Int)
    N = inst.N;  C = inst.C;  w = inst.w

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(model, z[1:N], Bin)
    @variable(model, 0 <= f[1:N+1] <= 1)

    @objective(model, Max,
        sum(w[i] * f[i] + dual_yopt[i, profile] * z[i] for i in 1:N))

    for i in 1:N
        @constraint(model, f[i] <= z[i] * v[i])
        @constraint(model, f[i] <= f[N+1] * v[i])
        @constraint(model, f[i] >= v[i] * (f[N+1] - (1 - z[i])))
    end

    @constraint(model, sum(f[i] for i in 1:N+1) == 1)
    @constraint(model, sum(z[i] for i in 1:N) <= C)

    optimize!(model)

    obj  = objective_value(model)
    zopt = round.(Int, value.(z))
    fopt = value.(f)

    return obj, zopt, fopt
end

# ─────────────────────────────────────────────────────────────────────────────
#  Primal LP
# ─────────────────────────────────────────────────────────────────────────────

"""
    RLB_Primal(inst, Nhat, Di, Ki, E1, E2) -> Matrix{Float64}

Solve the RLB primal LP after column generation.

Returns
-------
- `yopt` : Optimal y variables (Esize × M).
"""
function RLB_Primal(inst::InstanceData,
                    Nhat::Matrix{Int}, Di::Matrix{Int},
                    Ki::Matrix{Float64},
                    E1::Matrix{Int}, E2::Vector{Float64})
    N = inst.N;  M = inst.M
    Esize = size(E1, 2)

    model = Model(() -> Gurobi.Optimizer(GRB_ENV))

    @variable(model, x[1:N, 1:M] >= 0)
    @variable(model, y[1:Esize, 1:M] >= 0)

    @objective(model, Min, sum(E2[e] * y[e, E1[N+1, e]] for e in 1:Esize))

    for i in 1:N
        Nhat[i, 1] == 0 && continue
        @constraint(model, sum(x[i, j] * Di[i, j] for j in 1:M) >= Ki[i, 1])
    end

    for i in 1:N, j in 1:M
        e_idx = findall(==(j), E1[N+1, :])
        @constraint(model, sum(y[e, j] * E1[i, e] for e in e_idx) - x[i, j] >= 0)
    end

    optimize!(model)
    return value.(y)
end
