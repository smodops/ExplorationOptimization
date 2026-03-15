###############################################################################
#  TauSet.jl —  Macro-period schedule 
#
#  Generates the sequence of customer indices at which the ExpOpt and RLB policies
#  transition to the next macro period. The schedule grows exponentially with
#  base exp(1/H), ensuring the macro periods cover [1, T].
###############################################################################

"""
    TauSet(H, T) -> Vector{Int}

Compute the macro-period boundary schedule τ₁, τ₂, … used by the ExpOpt and RLB
policies.

Arguments
---------
- `H` : Exploration tuning parameter.
- `T` : Total number of customers (horizon).

Returns
-------
- Strictly increasing integer vector of macro-period boundaries.
"""
function TauSet(H::Int, T::Int)::Vector{Int}
    nmax    = Int(floor(H * log(T))) + 1
    tau_set = Vector{Int}(undef, nmax)

    for n in 1:nmax
        tau_temp = Int(ceil(exp(n / H)))
        if n > 1
            tau_set[n] = max(tau_temp, tau_set[n-1] + 1)
        else
            tau_set[n] = tau_temp
        end
    end

    return tau_set
end
