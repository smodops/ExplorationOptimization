###############################################################################
#  AssortmentPolicies.jl — Module entry point
#
#  Usage (from Main.jl):
#    include("src/AssortmentPolicies.jl")
#    using .AssortmentPolicies
#
#  Public API (exported symbols):
#    Types    : SimParams, InstanceData, InitData, TSPrior, PolicyResult
#    Policies : ExpOpt_Policy, RLB_Policy, TS_Policy
#    Helpers  : TauSet, AssortmentStatic, StaticSol, FeasibleAssortments, 
#               PlotResults, Greedy_MILP, ColGen, ExploreReq, Di_Analysis
###############################################################################

module AssortmentPolicies

using JuMP, Gurobi, Statistics, StatsBase, Optim,
      LinearAlgebra, Distributions, Random, Plots

# ── Gurobi environment (shared across all solver calls) ───────────────────────
const GRB_ENV = Gurobi.Env()
Gurobi.GRBsetintparam(GRB_ENV, "OutputFlag", 0)

# ── Exported symbols ──────────────────────────────────────────────────────────
export SimParams, InstanceData, InitData, TSPrior, PolicyResult
export ExpOpt_Policy, RLB_Policy, TS_Policy
export TauSet, AssortmentStatic, StaticSol, FeasibleAssortments
export PlotResults
export Greedy_MILP, ColGen, ExploreReq, Di_Analysis

# ── Source files (order matters: Types and Utils before everything else) ──────
include("Types.jl")
include("Utils.jl")
include("TauSet.jl")
include("FeasibleAssortments.jl")
include("AssortmentStatic.jl")
include("StaticSol.jl")
include("MLE.jl")
include("ExploreReq.jl")
include("Greedy_MILP.jl")
include("Di_Analysis.jl")
include("RLB_Solvers.jl")
include("ExpOpt_Policy.jl")
include("RLB_Policy.jl")
include("TS_Policy.jl")
include("PlotResults.jl")

end # module AssortmentPolicies
