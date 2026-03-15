###############################################################################
#  RLB_Policy.jl — RLB-Based Policy
#
#  Two entry points:
#    RLB_Policy(params, inst, tau_set)         — cold start
#    RLB_Policy(params, inst, tau_set, init)   — warm start (initialized)
#
#  Both dispatch to `_run_RLB`.
###############################################################################

"""
    RLB_Policy(params, inst, tau_set [, init]) -> PolicyResult

Run the RLB-Based Learning policy for `params.R` replications.

Pass an `InitData` object as the fourth argument for the warm-start variant.

Arguments
---------
- `params`  : `SimParams` — simulation hyper-parameters.
- `inst`    : `InstanceData` — instance data (includes S, Smax).
- `tau_set` : Macro-period boundary schedule (from `TauSet`).
- `init`    : Optional `InitData` for warm-start; `nothing` for cold start.

Returns
-------
- `PolicyResult` with policy's mean regret, its 95% confidence interval, and per-replication runtimes.
"""
function RLB_Policy(params::SimParams, inst::InstanceData,
                    tau_set::Vector{Int},
                    init::Union{InitData,Nothing}=nothing)::PolicyResult
    return _run_RLB(params, inst, tau_set, init)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Internal runner
# ─────────────────────────────────────────────────────────────────────────────

function _run_RLB(params::SimParams, inst::InstanceData,
                  tau_set::Vector{Int},
                  init::Union{InitData,Nothing})::PolicyResult
    T = params.T;  R = params.R;  Spac = params.Spac
    H = params.H;  MCU = params.MCU; C = inst.C
    N = inst.N;    M = inst.M;    D = inst.D
    Lngth = Int(ceil(T / Spac))

    J      = zeros(Lngth, R)
    time_r = zeros(R)

    for r in 1:R
        # ── Per-replication initialization ───────────────────────────────────
        T_init   = isnothing(init) ? 0 : init.T_init
        n        = isnothing(init) ? zeros(N, M) : reshape(copy(init.s), N, M)
        Et       = 0.0
        Er       = zeros(Lngth)

        hist_zt  = zeros(T, N+1)
        hist_st  = zeros(T, N)
        class    = zeros(T)

        t_rlb    = 0
        RLB_yopt = Matrix{Float64}(undef, 0, 0)
        Sexploit = zeros(N, M)
        f        = zeros(1, M)

        # E: initial assortment-profile matrix for column generation algorithm 
        E  = zeros(N+2, inst.Smax)
        E1 = Matrix{Int}(undef, N+1, 0)
        E2 = Vector{Float64}(undef, 0)

        exploration_set = Dict{Int, Matrix{Int}}()

        data   = Matrix{Float64}(undef, 0, N+2)
        assort = Matrix{Float64}(undef, 0, N)

        macro_n = 1
        start   = time()

        for t in 1:T
            # ── Update trigger ───────────────────────────────────────────────
            # Cold-start: trigger at exponential macro-period boundaries (tau_set)
            # Warm-start: trigger every MCU customers, counting from last solve
            triggered = isnothing(init) ? (t == tau_set[macro_n]) :
                                          (t - t_rlb >= MCU || t == 1)

            if triggered
                isnothing(init) && (macro_n += 1)

                # MLE
                vhat = zeros(N, M)
                try
                    # Assemble MLE data
                    if t == 1 && !isnothing(init)
                        data, assort = init.purchase, init.assort
                    else
                        data, assort = build_data(hist_zt, class, hist_st, t)
                        if !isnothing(init)
                            data, assort = prepend_init(data, assort, init)
                        end
                    end
                    
                    Beta_0 = fill(-1.0, N, D)
                    ll(B)  = loglike(B, inst, data, assort)
                    gr(B)  = grad(B, inst, data, assort)
                    res    = Optim.optimize(ll, gr, Beta_0, ConjugateGradient();
                                           inplace=false)
                    vhat   = exp.(Optim.minimizer(res) * inst.X)
                    for m in 1:M
                        f[m], Sexploit[:, m] = StaticSol(vhat[:, m], inst)
                    end
                    log_solved("RLB MLE", t)
                catch e
                    log_failed("RLB MLE", t, e)
                end

                # Column-Genration algorithm to solve RLB (cold-start: skip if solved recently; warm-start: always solve)
                if isnothing(init) ? (t - t_rlb > 20) : true
                    try
                        _ , Di, Nhat, Ki, E = Di_Analysis(vhat, inst, Sexploit, f)
                        Di   = Matrix{Int}(Di)
                        Nhat = Matrix{Int}(Nhat)
                        Ki   = Matrix{Float64}(Ki)
                        E1   = Matrix{Int}(round.(E[1:N+1, :]))
                        E2   = Vector{Float64}(E[N+2, :])

                        E1, E2   = ColGen(inst, Nhat, Di, Ki, E1, E2, vhat, f)
                        RLB_yopt = RLB_Primal(inst, Nhat, Di, Ki, E1, E2)

                        log_solved("RLB", t)
                        t_rlb = t
                    catch e
                        log_failed("RLB", t, e) # failure due to numerical issues is non-critical; will retry at next trigger
                    end

                end
            end

            # ── Customer arrival ─────────────────────────────────────────────
            It       = sample_profile(inst.pcum, M)
            class[t] = It

            # ── Assortment decision ──────────────────────────────────────────
            if t == 1 || isempty(RLB_yopt)
                St = random_assortment(N, C)
            else
                # Build exploration set for current profile
                try
                    e_idx = findall(>(0), RLB_yopt[:, It])
                    eset  = hcat(E1[1:N, e_idx], Sexploit[:, It])
                    exploration_set[It] = Matrix{Int}(round.(eset))
                catch
                    if !haskey(exploration_set, It) || isempty(exploration_set[It])
                        exploration_set[It] = reshape(Int.(Sexploit[:, It]), N, 1)
                    end
                end

                St = float(Sexploit[:, It])   # default: exploit
                eset = exploration_set[It]
                explore_threshold = H * log(t + T_init)
                for col in eachcol(eset)
                    if any((n[:, It] .< explore_threshold) .& (col .== 1.0))
                        St = float(col)
                        break
                    end
                end
            end

            # ── Purchase simulation ──────────────────────────────────────────
            rt, eSt = simulate_purchase(St, inst.eU[:, It], inst.w, N)
            Et += inst.fopt[It] - eSt
            n[:, It]   .+= vec(St)
            hist_zt[t, rt] = 1.0
            hist_st[t, :]  = vec(St)

            if mod(t, Spac) == 0
                Er[div(t, Spac)] = Et
            end
        end

        time_r[r] = time() - start
        J[:, r]   = Er
        log_replication(r, time_r[r])
    end

    Jm, Jlb, Jub = regret_ci(J)
    return PolicyResult(Jm, Jlb, Jub, time_r)
end
