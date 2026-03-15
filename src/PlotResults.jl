###############################################################################
#  PlotResults.jl — Plot cumulative regret for all three policies
#
#  Generates a single PDF figure showing the mean cumulative regret and 95%
#  confidence interval bands for ExpOpt, RLB, and TS policies. When warm-start
#  data is used, the regret accumulated during the initialization period is
#  added to all policy results so the plot reflects total regret from t = 0.
###############################################################################

"""
    PlotResults(res_ExpOpt, res_RLB, res_TS, inst, params, instance,
                save_path [, init, pur, asr])

Generate and save a regret comparison plot as a PDF.

When `init` is provided (warm-start case), the cumulative regret over the
initialization period is computed from `pur` and `asr` and added to all
policy results before plotting. The x-axis is shifted accordingly so it
reflects total customer arrivals (initialization + simulation).

Arguments
---------
- `res_ExpOpt` : `PolicyResult` from the ExpOpt policy.
- `res_RLB`    : `PolicyResult` from the RLB policy.
- `res_TS`     : `PolicyResult` from the TS policy.
- `inst`       : `InstanceData` struct.
- `params`     : `SimParams` struct.
- `instance`   : Instance identifier string (used in filename and title).
- `save_path`  : Directory where the PDF will be saved.
- `init`       : Optional `InitData`; pass `nothing` for cold-start instances.
- `pur`        : Purchase history matrix from initialization (required if init ≠ nothing).
- `asr`        : Assortment history matrix from initialization (required if init ≠ nothing).
"""
function PlotResults(res_ExpOpt::PolicyResult, res_RLB::PolicyResult, res_TS::PolicyResult,
                     inst::InstanceData, params::SimParams,
                     instance::String, save_path::String,
                     init::Union{InitData, Nothing} = nothing,
                     pur::Union{Matrix{Float64}, Nothing} = nothing,
                     asr::Union{Matrix{Float64}, Nothing} = nothing)

    T    = params.T
    Spac = params.Spac
    N    = inst.N

    # ── Compute initialization period regret ─────────────────────────────────
    if !isnothing(init) && !isnothing(pur) && !isnothing(asr)
        arrived =       Int.(vec(pur[:, N+2]))
        vt              = asr .* inst.eU[:, arrived]'
        pSt             = vt ./ (1.0 .+ sum(vt; dims=2))
        eSt             = pSt * inst.w[1:N]
        init_regret_sum = sum(vec(inst.fopt)[arrived] .- eSt)
        T_init          = size(pur, 1)
    else
        init_regret_sum = 0.0
        T_init          = 0
    end

    # ── Shift x-axis and add initialization regret to results ────────────────
    x_axis = collect(T_init + Spac : Spac : T_init + T)

    for res in [res_ExpOpt, res_RLB, res_TS]
        res.mean .+= init_regret_sum
        res.lb   .+= init_regret_sum
        res.ub   .+= init_regret_sum
    end

    # ── Build plot ────────────────────────────────────────────────────────────
    labels  = ["ExpOpt", "RLB", "Thompson Sampling"]
    colors  = [:red, :black, :blue]
    results = [res_ExpOpt, res_RLB, res_TS]

    plt = plot(
        size           = (900, 600),
        xlabel         = "T",
        ylabel         = "Regret",
        legend         = :topleft,
        grid           = true,
        gridalpha      = 0.3,
        framestyle     = :box,
        tickfontsize   = 14,
        guidefontsize  = 14,
        legendfontsize = 12,
        left_margin    = 12Plots.mm,
        right_margin   = 8Plots.mm,
        top_margin     = 8Plots.mm,
        bottom_margin  = 8Plots.mm,
    )

    for (res, label, color) in zip(results, labels, colors)
        # Shaded 95% CI band
        plot!(plt, x_axis, res.ub;
              fillrange = res.lb,
              fillalpha = 0.1,
              fillcolor = color,
              linewidth = 0,
              label     = "")
        # Mean regret line
        plot!(plt, x_axis, res.mean;
              color     = color,
              linewidth = 4,
              label     = label)
    end

    # ── Save ──────────────────────────────────────────────────────────────────
    plot_path = joinpath(save_path, "Regret_Instance$(instance).pdf")
    savefig(plt, plot_path)
    println("  Saved: $plot_path")

    return plt
end
