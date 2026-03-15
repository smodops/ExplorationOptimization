###############################################################################
#  FeasibleAssortments.jl — Enumerate all feasible assortments
#
#  Pre-computes the full set of feasible assortments once at startup.
#  The result is stored in InstanceData.S and used to handle possible numerical 
#  errors while running the column-generation algorithm for RLB policy.
###############################################################################

"""
    FeasibleAssortments(N, C) -> Matrix{Float64}

Enumerate all feasible assortments of at most `C` products out of `N`.

Each column of the returned matrix is a binary indicator vector of length `N`
representing one assortment. The empty assortment (all zeros) is excluded.

Arguments
---------
- `N` : Number of products.
- `C` : Maximum assortment size (capacity constraint).

Returns
-------
- `S` : Binary matrix of shape (N × Smax); each column is one assortment.
"""
function FeasibleAssortments(N::Int, C::Int)::Matrix{Float64}

    S_full = zeros(Float64, N, 2^N)
    for i in 1:N
        block = repeat([ones(Float64, 1, 2^(N-i))  zeros(Float64, 1, 2^(N-i))], 1, 2^(i-1))
        S_full[i, :] = vec(block)
    end

    # Keep only columns whose sum is in [1, C] (non-empty and within capacity)
    col_sums = vec(sum(S_full; dims=1))
    valid    = findall(s -> 1 <= s <= C, col_sums)

    return S_full[:, valid]
end
