###############################################################################
#  Di_Analysis.jl — RLB exploration structure analysis
#
#  Constructs the sets Oᵢ, Xᵢ, N̂ᵢ, Dᵢ, Kᵢ, and the initial column-generation
#  matrix E used by the RLB policy's dual problem.
###############################################################################

"""
    Di_Analysis(vhat, inst, Sexploit, f) -> (Di_profile, Di, Nhat, Ki, E)

Analyse the exploration structure required for the RLB policy.

Returns
-------
- `Di_profile` : (N × M) matrix; Di_profile[i, k] = profile index of the k-th
                 profile in Dᵢ.
- `Di`         : Binary (N × M) indicator of Dᵢ.
- `Nhat`       : (N × 3) matrix; column 1 = exploration flag, 2 = rank(Xᵢ),
                 3 = rank(Oᵢ).
- `Ki`         : (N × 1) vector; Kᵢ = 1/|Dᵢ| if |Dᵢ| > 0, else 0.
- `E`          : (N+2 × ?) initial assortment-profile matrix for column
                 generation algorithm; rows 1:N are assortment indicators, row N+1 is
                 the profile index, row N+2 is the optimality gap.

Arguments
---------
- `vhat`     : Estimated attractiveness matrix (N × M).
- `inst`     : `InstanceData` struct (N, M, C, X, w, Smax).
- `Sexploit` : Current exploitation assortments (N × M).
- `f`        : Current estimated optimal revenues (1 × M).
"""
function Di_Analysis(vhat::Matrix{Float64}, inst::InstanceData,
                     Sexploit::Matrix{Float64}, f::Matrix{Float64})
    N = inst.N;  M = inst.M;  C = inst.C
    X = inst.X;  w = inst.w

    # ── Step 1: Oᵢ — profiles for which product i is in the optimal assortment ──
    Oi = Int.(Sexploit .> 0)   # (N × M)

    # ── Step 2: Xᵢ — profiles where optimal revenue ≤ wᵢ (i.e., identify potentially optimal profiles) ──
    Xi = zeros(Int, N, M)
    for i in 1:N
        for m in 1:M
            Xi[i, m] = (f[m] <= w[i]) ? 1 : 0
        end
    end

    # ── Step 3: N̂ᵢ — exploration flag and ranks ─────────────────────────────
    Nhat = zeros(Int, N, 3)
    for i in 1:N
        oi_profiles = findall(>(0), Oi[i, :])
        xi_profiles = findall(>(0), Xi[i, :])
        r_Xi = isempty(xi_profiles) ? 0 : rank(X[:, xi_profiles])
        r_Oi = isempty(oi_profiles) ? 0 : rank(X[:, oi_profiles])
        Nhat[i, 2] = r_Xi                                       # rank of X(i)
        Nhat[i, 3] = r_Oi                                       # rank of O(i)
        Nhat[i, 1] = (r_Xi > r_Oi) ? 1 : 0
    end

    # ── Step 4: Dᵢ — informative profiles that expand rank beyond Oᵢ ─────────
    Di_profile = zeros(Int, N, M)
    Di         = zeros(Int, N, M)
    for i in 1:N
        Nhat[i, 1] == 0 && continue
        oi_profiles = findall(>(0), Oi[i, :])
        r_oi        = Nhat[i, 3]
        k = 0
        for m in 1:M
            in(m, oi_profiles) && continue
            if rank(hcat(X[:, oi_profiles], X[:, m])) == r_oi + 1
                k += 1
                Di_profile[i, k] = m
                Di[i, m]         = 1
            end
        end
    end

    # ── Step 5: Kᵢ = 1 / |Dᵢ| ───────────────────────────────────────────────
    Ki = zeros(N, 1)
    for i in 1:N
        d_size = sum(Di[i, :])
        Ki[i]  = (d_size > 0) ? 1.0 / d_size : 0.0
    end

    # ── Step 6: Build initial E matrix ───────────────────────────────────────
    # Rank products for each profile by descending estimated utility (attractiveness).
    class_top = [sortperm(vhat[:, m]; rev=true) for m in 1:M]

    cols = Vector{Vector{Float64}}()

    for m in 1:M
        # Products in Dᵢ that belong to profile m
        Em  = findall(>(0), Di[:, m])
        ems = length(Em)

        # Complement: top products not in Em, used to fill assortment to capacity
        cm = filter(p -> p ∉ Em, class_top[m])

        if ems == 0
            # No exploration products: just add the exploitation assortment
            col = zeros(N+2)
            col[N+1] = m
            top_c = Int.(cm[1:min(C, length(cm))])
            col[top_c] .= 1
            col[N+2] = compute_gap(col[1:N], vhat[:, m], w, f[m])
            push!(cols, col)
        elseif ems < C
            col = zeros(N+2)
            col[N+1] = m
            col[Em] .= 1
            ec = Int.(cm[1:C-ems])
            col[ec] .= 1
            col[N+2] = compute_gap(col[1:N], vhat[:, m], w, f[m])
            push!(cols, col)
        else
            # Split Em into chunks of size C
            ss = Int(ceil(ems / C))
            for s in 1:ss
                col = zeros(N+2)
                col[N+1] = m
                lo = (s-1)*C + 1
                hi = (s < ss) ? s*C : ems
                col[Em[lo:hi]] .= 1

                # Fill remaining slots in last chunk
                if s == ss && (ss*C - ems) > 0
                    empty_spots = ss*C - ems
                    cm_s = filter(p -> p ∉ Em[lo:hi], class_top[m])
                    col[Int.(cm_s[1:empty_spots])] .= 1
                end
                col[N+2] = compute_gap(col[1:N], vhat[:, m], w, f[m])
                push!(cols, col)
            end
        end
    end

    E = hcat(cols...)

    return Di_profile, Di, Nhat, Ki, E
end

# ─────────────────────────────────────────────────────────────────────────────
#  Internal helper: compute optimality gap for an assortment column
# ─────────────────────────────────────────────────────────────────────────────

function compute_gap(S_col::Vector{Float64}, v::Vector{Float64},
                     w::Vector{Float64}, f_m::Float64)::Float64
    v_s  = S_col .* v
    prob = v_s ./ (1.0 + sum(v_s))
    rev  = dot(w[1:length(v)], prob)
    return max(f_m - rev, 0.0)
end
