###############################################################################
#                                                                             #
#   Main Runner for experiments in Section 9.3 of:                            #
#   "Exploration Optimization for Dynamic Assortment Personalization under    #
#    Linear Preferences"                                                      #
#                                                                             #
#   Usage:                                                                    #
#     julia Run_scalability.jl                                                #
#                                                                             #
#  Measures running time for:                                                 #                                       
#    - Greedy-MILP        (src/Greedy_MILP.jl)                                #                  
#    - Column Generation  (src/RLB_Solvers.jl)                                #
#                                                                             #    
#  Experiment 1 — scale by number of profiles  (folders 0 ... ITER_PROFILES)  #
#  Experiment 2 — scale by number of products  (folders 1 ... ITER_PRODUCTS)  #
#                                                                             #
###############################################################################

using JuMP, Gurobi, Statistics, XLSX, DataFrames, LinearAlgebra

include("src/AssortmentPolicies.jl")
using .AssortmentPolicies

###############################################################################
#  USER SETTINGS
###############################################################################

PROFILE_PATH  = "./Data/Scalability/Profiles"
PRODUCT_PATH  = "./Data/Scalability/Products"
SAVE_PATH     = "./Results/Scalability"

ITER_PROFILES = 19      # sweep folders 0 ... 19
ITER_PRODUCTS = 2       # sweep folders 1 ... 2

C            = 4        # assortment capacity
MIP_TIME_LIM = 1800     # Gurobi time limit (seconds)
MIP_GAP      = 0.025    # Gurobi optimality gap---only applied in Experiment 2 (product scaling)
###############################################################################
#  DATA LOADING
###############################################################################

function load_sheet(path::String, file::String, DType::Type)
    wb = XLSX.readxlsx(joinpath(path, file))
    data = wb["Sheet1"][:]
    # Drop header row if first row contains strings
    if any(x -> x isa String, data[1, :])
        data = data[2:end, :]
    end
    return convert(Array{DType}, permutedims(data))
end

###############################################################################
#  BUILD InstanceData
###############################################################################

function build_inst(eU::Matrix{Float64}, X::Matrix{Float64}, N::Int, M::Int)
    w    = Vector{Float64}([ones(N); 0.0])
    p    = fill(1.0 / M, 1, M)
    pcum = cumsum(p; dims=2)
    S    = Matrix{Float64}(undef, N, 0)
    Smax = 0
    D    = size(X, 1)

    fopt = zeros(1, M)
    Sopt = zeros(N, M)
    for m in 1:M
        v                = eU[:, m]
        ix               = sortperm(v; rev=true)
        Sopt[ix[1:C], m] .= 1.0
        denom            = 1.0 + dot(v, Sopt[:, m])
        fopt[m]          = dot(v, Sopt[:, m]) / denom
    end

    return InstanceData(p, pcum, eU, X, N, C, D, M, w, S, Smax, fopt, Sopt)
end

function save_matrix(mat, folder::String, filename::String)
    df  = DataFrame(mat, :auto)
    out = joinpath(folder, filename)
    XLSX.writetable(out, collect(DataFrames.eachcol(df)), DataFrames.names(df); overwrite=true)
end

###############################################################################
#  EXPERIMENT RUNNERS
###############################################################################

function run_greedy_milp(folder::String, params::SimParams)::Float64
    eU   = load_sheet(folder, "eU.xlsx", Float64)   # N x M
    X    = load_sheet(folder, "X.xlsx",  Float64)   # D x M
    N, M = size(eU)
    inst = build_inst(eU, X, N, M)
    ExpReq  = ExploreReq(eU, inst, inst.Sopt, inst.fopt)

    println("  [Greedy-MILP] N=$N  M=$M")
    t0      = time()
    Greedy_MILP(eU, inst, ExpReq, params)
    elapsed = time() - t0
    println("  [Greedy-MILP] $(round(elapsed; digits=2)) s")
    return elapsed
end

function run_colgen(folder::String)::Float64
    eU   = load_sheet(folder, "eU.xlsx", Float64)   # N x M
    X    = load_sheet(folder, "X.xlsx",  Float64)   # D x M
    N, M = size(eU)
    inst = build_inst(eU, X, N, M)

    _, Di, Nhat, Ki, E = Di_Analysis(eU, inst, inst.Sopt, inst.fopt)
    Di   = Matrix{Int}(Di)
    Nhat = Matrix{Int}(Nhat)
    Ki   = Matrix{Float64}(Ki)
    E1   = Matrix{Int}(round.(E[1:N+1, :]))
    E2   = Vector{Float64}(E[N+2, :])

    println("  [ColGen] N=$N  M=$M ")
    t0      = time()
    ColGen(inst, Nhat, Di, Ki, E1, E2, eU, inst.fopt)
    elapsed = time() - t0
    println("  [ColGen] $(round(elapsed; digits=2)) s")
    return elapsed
end

###############################################################################
#  SHARED EXPERIMENT LOOP
###############################################################################

function run_experiment(root::String, indices, params::SimParams)
    rt_milp   = zeros(length(indices))
    rt_colgen = zeros(length(indices))

    for (k, it) in enumerate(indices)
        folder = joinpath(root, string(it))
        println("\n------------------------- Instance $it -------------------------")
        rt_milp[k]   = run_greedy_milp(folder, params)
        rt_colgen[k] = run_colgen(folder)
    end

    return DataFrame(instance=collect(indices), Greedy_MILP=rt_milp, ColGen=rt_colgen)
end

function save_results(df::DataFrame, filename::String)
    out = joinpath(SAVE_PATH, filename)
    XLSX.writetable(out, collect(DataFrames.eachcol(df)), DataFrames.names(df); overwrite=true)
    println("Saved: $out")
end

###############################################################################
#  MAIN
###############################################################################

# T, R, Spac, H, and MCU are placeholders -- only MIP_time_limit and MIP_gap matter here
params_profile = SimParams(0, 1, 1, 1, MIP_TIME_LIM, 0.0, 1)
params_product = SimParams(0, 1, 1, 1, MIP_TIME_LIM, MIP_GAP, 1)

println("\n#-------- Experiment 1: Scale by number of profiles --------#")
df1 = run_experiment(PROFILE_PATH, 0:ITER_PROFILES, params_profile)
save_results(df1, "run_times_scale_profiles.xlsx")

println("\n#-------- Experiment 2: Scale by number of products --------#")
df2 = run_experiment(PRODUCT_PATH, 0:ITER_PRODUCTS, params_product)
save_results(df2, "run_times_scale_products.xlsx")

println("\nDone!")
