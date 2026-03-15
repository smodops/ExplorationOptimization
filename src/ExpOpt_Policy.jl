###############################################################################
#  ExpOpt_Policy.jl — Exploration-Optimization Policy
#
#  Two entry points:
#    ExpOpt_Policy(params, inst, tau_set)             — cold start
#    ExpOpt_Policy(params, inst, tau_set, init)       — warm start (initialized)
#
#  Both dispatch to the same internal runner `_run_EO`, which accepts an
#  optional `InitData` argument.
###############################################################################

"""
    ExpOpt_Policy(params, inst, tau_set [, init]) -> PolicyResult

Run the Exploration-Optimization policy for `params.R` replications.

Pass an `InitData` object as the fifth argument to use the warm-start variant;
omit it (or pass `nothing`) for the cold-start variant.

Arguments
---------
- `params`   : `SimParams` — simulation hyper-parameters.
- `inst`     : `InstanceData` — instance data.
- `tau_set`  : Macro-period schedule (from `TauSet`).
- `init`     : Optional `InitData` for warm-start; `nothing` for cold start.

Returns
-------
- `PolicyResult` with policy's mean regret, its 95% confidence interval, and per-replication runtimes.
"""
function ExpOpt_Policy(params::SimParams, inst::InstanceData, tau_set::Vector{Int},
                   init::Union{InitData,Nothing}=nothing)::PolicyResult
    return _run_EO(params, inst, tau_set, init)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Internal runner
# ─────────────────────────────────────────────────────────────────────────────

function _run_EO(params::SimParams, inst::InstanceData, tau_set::Vector{Int},
                 init::Union{InitData,Nothing})::PolicyResult
    T = params.T;  R = params.R;  Spac = params.Spac
    H = params.H;  MCU = params.MCU; C = inst.C
    N = inst.N;    M = inst.M;    D = inst.D

    AC = Int(ceil(N / C))         # number of assortment copies per profile in the MILP formulation
    Lngth = Int(ceil(T / Spac))   # number of reporting intervals

    J      = zeros(Lngth, R)
    time_r = zeros(R)

    for r in 1:R
        # ── Per-replication initialization ───────────────────────────────────
        T_init   = isnothing(init) ? 0 : init.T_init
        n        = isnothing(init) ? zeros(N, M) : reshape(copy(init.s), N, M)
        Et       = 0.0
        Er       = zeros(Lngth)

        hist_zt  = zeros(T, N+1)   # purchase history
        hist_st  = zeros(T, N)     # assortment history
        class    = zeros(T)        # profile arrival history

        t_mip    = 0               # last period at which MILP was solved
        zxp      = []              # MILP solution (assortment copies)
        Sexploit = zeros(N, M)     # exploitation assortments
        f        = zeros(1, M)     # estimated optimal revenues

        # Shared MLE state (module-level for Optim closure compatibility)
        data   = Matrix{Float64}(undef, 0, N+2)
        assort = Matrix{Float64}(undef, 0, N)

        # ── Macro-period control ─────────────────────────────────────────────
        macro_n  = 1
        start    = time()

        # ── Main simulation loop ─────────────────────────────────────────────
        for t in 1:T

            # ── Macro-period or scheduled update trigger ─────────────────────
            # Cold-start: trigger at exponential macro-period boundaries (tau_set)
            # Warm-start: trigger every MCU customers, counting from last solve
            triggered = isnothing(init) ? (t == tau_set[macro_n]) :
                                          (t - t_mip >= MCU || t == 1)

            if triggered
                isnothing(init) && (macro_n += 1)

                # MLE
                vhat = zeros(N, M)
                try
                    # Assemble data for MLE
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
                    log_solved("ExpOpt MLE", t)
                catch e
                    log_failed("ExpOpt MLE", t, e)
                end

                # Exploration requirements
                ExpReq = ExploreReq(vhat, inst, Sexploit, f)

                # Greedy-MILP (cold-start: skip if solved recently; warm-start: always solve)
                if isnothing(init) ? (t - t_mip > 20) : true
                    try
                        zxp   = Greedy_MILP(vhat, inst, ExpReq, params)
                        t_mip = t  # update only if MILP succeeded (covers both warm and cold start)
                        log_solved("ExpOpt MILP", t)
                    catch e
                        log_failed("ExpOpt MILP", t, e) # failure due to numerical issues is non-critical; will retry at next trigger
                    end
                end
            end

            # ── Customer arrival ─────────────────────────────────────────────
            It  = sample_profile(inst.pcum, M)
            class[t] = It

            # ── Assortment decision ──────────────────────────────────────────
            if t == 1 || isempty(zxp)
                St = random_assortment(N, C)
            else
                St  = Sexploit[:, It]   # default: exploit
                zt  = zxp[:, It, :]     # assortment copies for profile It
                explore_threshold = H * log(t + T_init)

                for k in 1:AC
                    candidate = zt[:, k]
                    if any((n[:, It] .< explore_threshold) .& (candidate .== 1.0))
                        St = float(candidate)
                        break
                    end
                end
            end

            # ── Purchase simulation ──────────────────────────────────────────
            rt, eSt = simulate_purchase(St, inst.eU[:, It], inst.w, N)
            Et  += inst.fopt[It] - eSt
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
