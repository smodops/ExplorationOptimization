###############################################################################
#  ExploreReq.jl — Greedy-heuristic to identify the exploration requirements 
#  (used in the Exploration-Optimization MILP)
#
#  For each product i, determines which customer profiles must be explored
#  beyond the current exploitation set in order to achieve full-rank coverage
#  of the feature space. Returns a binary (N × M) matrix.
###############################################################################

"""
    ExploreReq(vhat, inst, Sexploit, f) -> Matrix{Float64}

Compute the binary exploration-requirement matrix `ExpReq` (N × M) for the ExpOpt
policy.

`ExpReq[i, j] = 1` means product i must be included in at least one assortment
offered to customers of profile j during the exploration phase.

For each product i, the heuristic greedily augments the set of profiles whose exploitation 
assortment includes product i, until the feature matrix X has full column rank over that set.

Arguments
---------
- `vhat`     : Estimated attractiveness matrix (N × M).
- `inst`     : `InstanceData` struct (N, M, X).
- `Sexploit` : Current exploitation assortments (N × M).
- `f`        : Current estimated optimal revenues (1 × M).

Returns
-------
- `ExpReq` : Binary (N × M) exploration requirement matrix.
"""

function ExploreReq(vhat::Matrix{Float64}, inst::InstanceData,
                    Sexploit::Matrix{Float64}, f::Matrix{Float64})::Matrix{Float64}
    N = inst.N;  M = inst.M;  X = inst.X
    full_rank = rank(X)

    # For each product i, rank profiles by descending estimated attractiveness vhat[i,:]
    profile_order = [sortperm(vhat[i, :]; rev=true) for i in 1:N]

    ExpReq = zeros(N, M)

    for i in 1:N
        # Profiles whose exploitation assortment includes product i
        exploit_profiles = findall(>(0), Sexploit[i, :])
        ExpReq[i, exploit_profiles] .= 1.0

        current_rank = rank(X[:, exploit_profiles])
        current_set  = copy(exploit_profiles)

        current_rank >= full_rank && continue   # already full rank — no extra exploration needed

        # Greedily add profiles in order of decreasing attractiveness until full rank
        candidate_profiles = setdiff(profile_order[i], exploit_profiles)
        for m in candidate_profiles
            if rank(hcat(X[:, current_set], X[:, m])) == current_rank + 1
                current_rank += 1
                push!(current_set, m)
                ExpReq[i, m] = 1.0
            end
            current_rank >= full_rank && break
        end
    end

    return ExpReq
end
