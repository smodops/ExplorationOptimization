###############################################################################
#                                                                             #
#   Main Simulation Runner for experiments in Section 9.2 of:                 #
#   "Exploration Optimization for Dynamic Assortment Personalization under    #
#    Linear Preferences"                                                      #
#                                                                             #
#   Usage:                                                                    #
#     julia Main.jl <instance_number>                                         #
#                                                                             #
#   Example:                                                                  #
#     julia Main.jl 1                                                         #
#                                                                             #
#   Per-instance parameters (T, R) are defined in Section 3 below.            #
#   Place all instance data in:  <base_read_path>/<instance>/                 #
#   Results are written to:      <base_save_path>/<instance>/                 #
#                                                                             #
###############################################################################

using XLSX, DataFrames, Printf, Random, Statistics

include("src/AssortmentPolicies.jl")
using .AssortmentPolicies

###############################################################################
# 1. PARSE AND VALIDATE INSTANCE NUMBER
###############################################################################
if length(ARGS) < 1
    error("""
    -----------------------------------------------------------------------
    ERROR: No instance number provided.
    Usage  : julia Main.jl <instance_number>
    Example: julia Main.jl 1
    -----------------------------------------------------------------------
    """)
end

instance = ARGS[1]
if isnothing(tryparse(Int, instance)) || parse(Int, instance) <= 0
    error("Instance number must be a positive integer. Got: \"$instance\"")
end
instance_id = parse(Int, instance)

println("=" ^ 70)
println("  Assortment Policy Simulation — Instance: $instance")
println("=" ^ 70)

###############################################################################
# 2. PATHS
#    *** Set base_read_path and base_save_path before running ***
###############################################################################
base_read_path = "./Data"    # e.g. "/Users/username/Data"
base_save_path = "./Results"    # e.g. "/Users/username/Results"

read_path = joinpath(base_read_path, instance)
save_path = joinpath(base_save_path, instance)
mkpath(save_path)

println("  Data path : $read_path")
println("  Save path : $save_path\n")

###############################################################################
# 3. PER-INSTANCE PARAMETERS
#  Instance 1 corresponds to left panel of Figure 2 in the paper
#  Instance 2 corresponds to right panel of Figure 2 in the paper
#  Instance 3 corresponds to left panel of Figure 3 in the paper
#  Instance 4 corresponds to right panel of Figure 3 in the paper
###############################################################################

# T = number of customers
# R = number of simulation replications for ExpOpt and RLB policies
# R_TS = number of simulation replications for Thompson Sampling policy
# initial_ind = 0 for no warm-start; 1 for warm-start 
# MCU = frequency of MLE/MILP updates

instance_params = Dict(
    1 => (T = 10000,  R = 40,  R_TS = 20,  initial_ind = 0, MCU = 277),
    2 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 1000),
    3 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 2000),
    4 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 2000),
)

if !haskey(instance_params, instance_id)
    error("""
    -----------------------------------------------------------------------
    ERROR: Instance $instance not found in instance_params.
    Valid instances: $(sort(collect(keys(instance_params))))
    -----------------------------------------------------------------------
    """)
end

T           = instance_params[instance_id].T
R           = instance_params[instance_id].R
R_TS        = instance_params[instance_id].R_TS
initial_ind = instance_params[instance_id].initial_ind
MCU         = instance_params[instance_id].MCU

println("  Instance $instance — T = $T customers, R = $R replications, R_TS = $R_TS, warm-start = $initial_ind\n")

###############################################################################
# 4. LOAD INSTANCE DATA
###############################################################################
println("  Loading input data...")

function load_sheet(path::String, file::String, DType::Type)
    wb = XLSX.readxlsx(joinpath(path, file))
    data = wb["Sheet1"][:]
    # Drop header row if first row contains strings
    if any(x -> x isa String, data[1, :])
        data = data[2:end, :]
    end
    return convert(Array{DType}, permutedims(data))
end

# p: customer profile arrival probabilities (M x 1)
# eU: exponentiated mean utilities (attractiveness) of products for customer profiles (N x M)
# X: customer profile features (D x M)
# N: number of products
# D: number of customer features
# M: number of customer profiles    
# C: assortment capacity constraint
# w: product revenues (N x 1, with no-purchase = 0)

p    = load_sheet(read_path, "Lambda.xlsx", Float64)
eU   = load_sheet(read_path, "exp_mean_utilities.xlsx", Float64)
X    = load_sheet(read_path, "X.xlsx", Float64)
pcum = cumsum(p; dims=2)

N    = size(eU, 1)
D    = size(X,  1)
M    = size(eU, 2)
C    = 4                                       
w    = Vector{Float64}([ones(N); 0.0])         

# Pre-enumerate all feasible assortments
S    = Matrix{Float64}(FeasibleAssortments(N, C))
Smax = size(S, 2)

println("  N=$N products  |  C=$C capacity  |  M=$M profiles  |  D=$D features")

###############################################################################
# 5. COMPUTE OFFLINE SOLUTIONS (optimal static assortment per profile)
###############################################################################
println("  Computing static optimal assortments...")

fopt     = zeros(1, M)
Sopt     = zeros(N, M)
inst_tmp = InstanceData(p, pcum, eU, X, N, C, D, M, w, S, Smax, fopt, Sopt)
for m in 1:M
    fopt[m], Sopt[:, m] = AssortmentStatic(eU[:, m], inst_tmp)
end

inst = InstanceData(p, pcum, eU, X, N, C, D, M, w, S, Smax, fopt, Sopt)

###############################################################################
# 6. SIMULATION PARAMETERS
#    T, R, and initial_ind come from the dictionary in Section 3.
#    All other parameters are shared across instances.
###############################################################################
Random.seed!(7)         # Simulation random seed (for reproducibility)

Spac           = 100    # reporting interval (every Spac customers)
H              = 4      # exploration tuning parameter
MIP_time_limit = 1800   # MILP solver time limit (seconds)
MIP_gap        = 0.00   # MILP optimality gap 

params    = SimParams(T, R,    Spac, H, MIP_time_limit, MIP_gap, MCU)
params_TS = SimParams(T, R_TS, Spac, H, MIP_time_limit, MIP_gap, MCU)

tau_set = TauSet(params.H, params.T)

# Thompson Sampling prior
sigma = fill(0.2, N*D, N*D)
for i in 1:N*D; sigma[i, i] = 1.0; end
ts_prior = TSPrior(fill(-1.0, N*D), sigma)

###############################################################################
# 7. LOAD WARM-START DATA (if applicable)
###############################################################################

#T_init: number of customers in the initialization data (used for warm-starting MLE/MILP)
# pur:  one-hot encoded purchase history from the initialization data (T_init x N+2 columns: N products + no-purchase option + profile ID)
# asr:  one-hot encoded assortment history from the initialization data (T_init x N columns: N products)
# s0:   display numbers of products to profiles in the initialization data (N x M)

init = nothing
if initial_ind == 1
    println("  Loading warm-start data...")
    pur  = convert(Matrix{Float64}, XLSX.readxlsx(joinpath(read_path, "purchase_history_initialization.xlsx"))["Sheet1"][:])
    asr  = convert(Matrix{Float64}, XLSX.readxlsx(joinpath(read_path, "assortment_history_initialization.xlsx"))["Sheet1"][:])
    s0   = reshape(load_sheet(read_path, "display_numbers_initialization.xlsx", Int), N, M)
    init = InitData(pur, asr, s0, size(pur, 1))
end

###############################################################################
# 8. HELPER — SAVE RESULTS
###############################################################################
function save_results(res::PolicyResult, policy_name::String)
    # Save regret confidence interval
    df  = DataFrames.DataFrame(LB=res.lb, Mean=res.mean, UB=res.ub)
    out = joinpath(save_path, "$(policy_name)_Regret_CI.xlsx")
    XLSX.writetable(out, collect(DataFrames.eachcol(df)), DataFrames.names(df); overwrite=true)
    println("  Saved: $out")

    # Save per-replication runtimes
    df_time  = DataFrames.DataFrame(RunTime=vec(res.time_r))
    out_time = joinpath(save_path, "$(policy_name)_RunTime.xlsx")
    XLSX.writetable(out_time, collect(DataFrames.eachcol(df_time)), DataFrames.names(df_time); overwrite=true)
    println("  Saved: $out_time")
end

###############################################################################
# 9. RUN POLICIES
###############################################################################

println("\n" * "─"^20 * " Running Exploration-Optimization Policy " * "─"^20)
res_ExpOpt  = ExpOpt_Policy(params, inst, tau_set, init)
save_results(res_ExpOpt, "ExpOpt")

println("\n" * "─"^20 * " Running RLB Policy " * "─"^20)
res_RLB = RLB_Policy(params, inst, tau_set, init)
save_results(res_RLB, "RLB")

println("\n" * "─"^20 * " Running Thompson Sampling Policy " * "─"^20)
res_TS  = TS_Policy(params_TS, inst, ts_prior, init)
save_results(res_TS, "TS")

###############################################################################
# 10. SUMMARY
###############################################################################
println()
println("=" ^ 70)
println("  SIMULATION COMPLETE — Instance $instance  (T=$T, R=$R, R_TS=$R_TS)")
println("  Output written to: $save_path")
println("=" ^ 70)
###############################################################################
# 11. PLOT RESULTS
###############################################################################
println("\n  Generating regret plot...")

PlotResults(res_ExpOpt, res_RLB, res_TS, inst, params, instance, save_path,
            init,
            initial_ind == 1 ? pur : nothing,
            initial_ind == 1 ? asr : nothing)

println("=" ^ 70)
