###############################################################################
#  Utils.jl — Shared utility functions for AssortmentPolicies
#
#  Functions here are used by multiple policy files. Centralising them avoids
#  code duplication and makes future changes easy to propagate.
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
#  Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    log_solved(label, t)

Print a standardised "solved" status message.
"""
function log_solved(label::String, t::Int)
    println("\n[✓] $label solved at customer t = $t\n")
end

"""
    log_failed(label, t, err)

Print a standardised "failed" status message including the error.
"""
function log_failed(label::String, t::Int, err)
    println("\n[✗] $label FAILED at customer t = $t — error: $err\n")
end

"""
    log_replication(r, t)

Print a standardised end-of-replication banner.
"""
function log_replication(r::Int, t::Float64)
    println("\n" * "─"^60)
    println("  END OF REPLICATION $r (Runtime: $t seconds)")
    println("─"^60 * "\n")
end

# ─────────────────────────────────────────────────────────────────────────────
#  Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_profile(pcum, M) -> Int

Draw a customer profile index by inverse transform sampling from the cumulative
arrival probability vector `pcum` (length M).

Note: the fallback to `M` handles the rare case where floating-point rounding
leaves the final cumulative probability just below 1.0 and `rand()` exceeds it.
"""
function sample_profile(pcum::Matrix{Float64}, M::Int)::Int
    rr  = rand()
    idx = findfirst(x -> x >= rr, view(pcum, 1, :))
    return isnothing(idx) ? M : idx
end

"""
    random_assortment(N, C) -> Vector{Int}

Return a random binary assortment vector of length N with exactly C ones.
"""
function random_assortment(N::Int, C::Int)::Vector{Float64}
    St = zeros(N)
    St[sample(1:N, C; replace=false)] .= 1.0
    return St
end

"""
    simulate_purchase(St, eU_col, w, N) -> (Int, Float64, Float64)

Given offered assortment `St` and true utilities `eU_col` (length N) for the
arriving customer's profile, simulate the purchase outcome.

Returns
-------
- `rt`  : Index of chosen product (N+1 = no purchase).
- `eSt` : Expected revenue of the offered assortment.
- `pSt` : Cumulative purchase probability vector (length N+1).
"""
function simulate_purchase(St::Vector{Float64}, eU_col::Vector{Float64},
                           w::Vector{Float64}, N::Int)
    vt   = St .* eU_col
    prob = vt ./ (1.0 + sum(vt))
    eSt  = dot(w[1:N], prob)
    cum  = [cumsum(prob); 1.0]
    rr   = rand()
    rt   = findfirst(x -> x >= rr, cum)
    return rt, eSt
end

# ─────────────────────────────────────────────────────────────────────────────
#  MLE helpers (shared across ExpOpt, RLB, TS policies)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_data(zt, class, st, t) -> (Matrix, Matrix)

Construct the `(data, assort)` matrices from simulation history up to period t-1.
"""
function build_data(zt::Matrix{Float64}, class::Vector{Float64},
                    st::Matrix{Float64}, t::Int)
    data   = [zt[1:t-1, :]   reshape(class[1:t-1], :, 1)]
    assort = st[1:t-1, :]
    return data, assort
end

"""
    prepend_init(data, assort, init::InitData) -> (Matrix, Matrix)

Prepend the warm-start initialization data to the current simulation data.
"""
function prepend_init(data::Matrix, assort::Matrix, init::InitData)
    return [Float64.(init.purchase); data], [Float64.(init.assort); assort]
end

# ─────────────────────────────────────────────────────────────────────────────
#  Statistics helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    regret_ci(J) -> (mean_vec, lb_vec, ub_vec)

Compute the mean and 95% confidence interval of regret across replications.
`J` is a (Lngth × R) matrix of regret trajectories.
"""
function regret_ci(J::Matrix{Float64})
    R    = size(J, 2)
    Jm   = vec(mean(J; dims=2))
    Jstd = vec(std(J;  dims=2))
    Jlb  = Jm .- 1.98 .* Jstd ./ sqrt(R)
    Jub  = Jm .+ 1.98 .* Jstd ./ sqrt(R)
    return Jm, Jlb, Jub
end
