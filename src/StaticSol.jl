################################################################################
#  StaticSol.jl — Solves the static assortment problem vie enumeration
#
#  Used by the RLB policy as an alternative to AssortmentStatic (MILP) to handle 
#  possible numerical errors while running the column-generation algorithm. 
################################################################################

"""
    StaticSol(v, inst) -> (Float64, Vector)

Arguments
---------
- `v`    : Product exponentiated mean utility (attractiveness) vector for the profile (length N).
- `inst` : `InstanceData` struct carrying N, w, and S.

Returns
-------
- Optimal expected revenue (Float64).
- Optimal assortment column from S (Vector, length N).
"""
function StaticSol(v::Vector{Float64}, inst::InstanceData)
    N = inst.N;  w = inst.w;  S = inst.S

    denom  = vec(v' * S) .+ 1.0          # 1 + Σᵢ vᵢ sᵢ  for each assortment
    numer  = vec((v .* w[1:N])' * S)     # Σᵢ wᵢ vᵢ sᵢ   for each assortment
    profit = numer ./ denom

    idx = argmax(profit)
    return profit[idx], S[:, idx]
end
