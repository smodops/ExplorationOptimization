###############################################################################
#  Types.jl — Shared data structures for AssortmentPolicies
###############################################################################

"""
    SimParams

Simulation-level hyper-parameters that are fixed for an entire experimental run.

Fields
------
- `T`              : Total number of arriving customers.
- `R`              : Number of simulation replications.
- `Spac`           : Reporting interval (regret is recorded every `Spac` steps).
- `H`              : Exploration tuning parameter.
- `MIP_time_limit` : Optimizer time limit (seconds) for MILP.
- `MIP_gap`        : MILP optimality gap tolerance for termination.
- `MCU`            : MLE/MILP re-solve frequency.
"""
struct SimParams
    T              :: Int
    R              :: Int
    Spac           :: Int
    H              :: Int
    MIP_time_limit :: Int
    MIP_gap        :: Float64
    MCU            :: Int
end

"""
    InstanceData

All instance-specific data loaded from disk for a single problem instance.

Fields
------
- `p`    : Customer profile arrival probability distribution (1 × M).
- `pcum` : Cumulative arrival probabilities (1 × M), used for fast sampling.
- `eU`   : Products' exponentiated mean utility (attractiveness) matrix (N × M).
- `X`    : Customer feature matrix (D × M).
- `N`    : Number of products.
- `C`    : Assortment capacity constraint.
- `D`    : Number of customer features.
- `M`    : Number of customer profiles.
- `w`    : Revenue vector (length N+1; last entry is no-purchase = 0).
- `S`    : Matrix of all feasible assortments (N × Smax).
- `Smax` : Total number of feasible assortments.
- `fopt` : Optimal static expected revenue per profile (1 × M).
- `Sopt` : Optimal static assortment per profile (N × M).
"""
struct InstanceData
    p    :: Matrix{Float64}
    pcum :: Matrix{Float64}
    eU   :: Matrix{Float64}
    X    :: Matrix{Float64}
    N    :: Int
    C    :: Int
    D    :: Int
    M    :: Int
    w    :: Vector{Float64}
    S    :: Matrix{Float64}
    Smax :: Int
    fopt :: Matrix{Float64}
    Sopt :: Matrix{Float64}
end

"""
    InitData

Warm-start data used by the `_Initialized` policy variants.

Fields
------
- `purchase` : Historical purchase decisions from the initialization period.
- `assort`   : Historical offered assortments from the initialization period.
- `s`        : Product exposure counts accumulated during initialization (N × M).
- `T_init`   : Length of the initialization period.
"""
struct InitData
    purchase :: Matrix{Float64}
    assort   :: Matrix{Float64}
    s        :: Matrix{Float64}
    T_init   :: Int
end

"""
    TSPrior

Prior distribution parameters for the Thompson Sampling policy.

Fields
------
- `mu`    : Prior mean vector (length N*D).
- `sigma` : Prior covariance matrix (N*D × N*D).
"""
struct TSPrior
    mu    :: Vector{Float64}
    sigma :: Matrix{Float64}
end

"""
    PolicyResult

Standardized container for the output of any policy run.

Fields
------
- `mean`   : Mean cumulative regret over replications at each report point.
- `lb`     : Lower bound of the 95% confidence interval.
- `ub`     : Upper bound of the 95% confidence interval.
- `time_r` : Runtime (seconds) for each replication.
"""
mutable struct PolicyResult
    mean   :: Vector{Float64}
    lb     :: Vector{Float64}
    ub     :: Vector{Float64}
    time_r :: Vector{Float64}
end
