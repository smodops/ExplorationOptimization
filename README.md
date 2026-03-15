# Replication Code

**"Exploration Optimization for Dynamic Assortment Personalization under Linear Preferences"**

This repository contains the replication code for the numerical experiments in Section 9.2 and 9.3 of the paper. It implements and compares three online learning policies for dynamic assortment personalization under the Multinomial Logit (MNL) choice model:

- **ExpOpt** — Exploration-Optimization policy
- **RLB** — RLB-based policy
- **TS** — Thompson Sampling policy

---

## Requirements

- Julia 1.9 or later
- Gurobi (with a valid license)

The following Julia packages are required. Install them once by running `] add <PackageName>` in the Julia REPL:

```
JuMP, Gurobi, XLSX, DataFrames, Optim, Distributions,
StatsBase, Statistics, Plots, Printf, Random, LinearAlgebra
```

---

## Project Structure

```
Main.jl                        ← main runner file for the 4 numerical experiments in Section 9.2 
Run_all_instances.jl           ← convenience script to run all 4 instances (Section 9.2) sequentially
Run_scalability.jl             ← scalability experiment runner (Section 9.3)
src/
    AssortmentPolicies.jl      ← module entry point
    Types.jl                   ← shared structs (SimParams, InstanceData, etc.)
    Utils.jl                   ← shared helper functions
    MLE.jl                     ← MNL log-likelihood and gradient functions 
    TauSet.jl                  ← macro-period schedule
    FeasibleAssortments.jl     ← enumerate all feasible assortments
    AssortmentStatic.jl        ← MILP for optimal static assortment
    StaticSol.jl               ← enumeration-based static assortment solver
    ExploreReq.jl              ← exploration requirement heuristic
    Greedy_MILP.jl             ← Exploration-Optimization MILP
    Di_Analysis.jl             ← RLB exploration structure analysis
    RLB_Solvers.jl             ← RLB dual, primal, subproblem, and column generation
    ExpOpt_Policy.jl           ← Exploration-Optimization policy
    RLB_Policy.jl              ← RLB-based policy
    TS_Policy.jl               ← Thompson Sampling policy
    PlotResults.jl             ← regret plot generation
```

---

## Performance Comparison Experiments (Section 9.2 of the paper)

### Data

Each instance requires its own subfolder under `Data/`. Place the following Excel files in `Data/<instance_number>/`:

| File | Description | Dimensions |
|---|---|---|
| `Lambda.xlsx` | Customer profile arrival probabilities | M × 1 |
| `exp_mean_utilities.xlsx` | Exponentiated mean utilities (attractiveness) of products for customer profiles | M × N |
| `X.xlsx` | Customer feature matrix | M × D |

For warm-start instances (instances 1–3), there is an initialization phase where a sample of 10,000 random transaction data is generated based on the ground truth and given to each policy:

| File | Description | Dimensions |
|---|---|---|
| `purchase_history_initialization.xlsx` | One-hot encoded purchase history from initialization period | T_init × (N+2) |
| `assortment_history_initialization.xlsx` | One-hot encoded assortment history from initialization period | T_init × N |
| `display_numbers_initialization.xlsx` | Product display counts per profile from initialization period | N × M |

---

### Configuration

Before running, set the data and results paths in `Main.jl` (Section 2):

```julia
base_read_path = "./Data"      # folder containing instance subfolders
base_save_path = "./Results"   # folder where results will be written
```

Per-instance simulation parameters are defined in the dictionary in Section 3 of `Main.jl`:

```julia
instance_params = Dict(
    1 => (T = 10000,  R = 40,  R_TS = 20,  initial_ind = 0, MCU = 277),
    2 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 1000),
    3 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 2000),
    4 => (T = 10000,  R = 20,  R_TS = 10,  initial_ind = 1, MCU = 2000),
)
```

| Parameter | Description |
|---|---|
| `T` | Number of customer arrivals in the simulation |
| `R` | Number of replications for ExpOpt and RLB policies |
| `R_TS` | Number of replications for the TS policy (fewer due to higher computational cost) |
| `initial_ind` | `1` = warm-start using initialization data; `0` = cold-start | 
| `MCU` | Frequency of MLE/MILP updates |

The four instances correspond to the figures in the paper:

| Instance | Figure |
|---|---|
| 1 | Left panel of Figure 2 |
| 2 | Right panel of Figure 2 |
| 3 | Left panel of Figure 3 |
| 4 | Right panel of Figure 3 |

---

### Running

Navigate to the project folder and run:

```bash
cd /path/to/project
julia Main.jl <instance_number>
```

For example, to run instance 1:

```bash
julia Main.jl 1
```

To run all four instances sequentially:

```bash
julia Main.jl 1 && julia Main.jl 2 && julia Main.jl 3 && julia Main.jl 4
```

To run a single instance and save the terminal output to a log file simultaneously (Mac/Linux):

```bash
mkdir -p logs
julia Main.jl 1 | tee logs/instance_1.log
```

On Windows:

```powershell
New-Item -ItemType Directory -Force -Path logs
julia Main.jl 1 | Tee-Object logs/instance_1.log
```

Alternatively, use the convenience script which runs all four instances automatically, reports timing for each, and handles logging automatically:

```bash
julia Run_all_instances.jl
```

Each instance is launched as a separate Julia process, ensuring a clean compilation state between runs. Terminal output for each instance is automatically saved to a timestamped `.log` file in the `logs/` folder:

---

### Output

All results are written to `Results/<instance_number>/`:

| File | Description |
|---|---|
| `ExpOpt_Regret_CI.xlsx` | ExpOpt policy cumulative regret: lower bound, mean, upper bound |
| `RLB_Regret_CI.xlsx` | RLB policy cumulative regret: lower bound, mean, upper bound |
| `TS_Regret_CI.xlsx` | TS policy cumulative regret: lower bound, mean, upper bound |
| `Regret_Instance<N>.pdf` | Regret comparison plot for all three policies |

Each Excel file has one row per reporting interval (`Spac = 100` customers) with three columns: `LB` (95% CI lower bound), `Mean`, and `UB` (95% CI upper bound).

The PDF plot shows the mean cumulative regret and shaded 95% confidence interval bands for all three policies on a single figure. For warm-start instances, the regret accumulated during the initialization period is included so the plot reflects total regret from the first customer onward.

---

## Scalability Experiments (Section 9.3 of the paper)

`Run_scalability.jl` measures the running time of the two core optimization solvers:

- **Greedy-MILP** (`src/Greedy_MILP.jl`) — Exploration-Optimization MILP
- **Column Generation** (`src/RLB_Solvers.jl`) — RLB column generation solver

Two experiments are run (see Section 9.3 of the paper for details):

| Experiment | Description |
|---|---|
| Experiment 1 | Scales by number of **customer profiles** (sweeps folders `0 … ITER_PROFILES`) |
| Experiment 2 | Scales by number of **products** (sweeps folders `1 … ITER_PRODUCTS`) |

### Data

Scalability instances are stored separately from the main simulation data. Place instance subfolders at the following paths:

```
Data/Scalability/Profiles/0/    ← profile-scaling instances (folders 0 … 19 by default)
Data/Scalability/Profiles/1/
...
Data/Scalability/Products/0/    ← product-scaling instances (folders 0 … 2 by default)
...
```

Each subfolder must contain two Excel files:

| File | Description | Dimensions |
|---|---|---|
| `eU.xlsx` | Exponentiated mean utilities (attractiveness) of products for customer profiles | M × N |
| `X.xlsx` | Customer feature matrix | M × D |

### Configuration

Key settings are defined at the top of `Run_scalability.jl`:

```julia
PROFILE_PATH  = "./Data/Scalability/Profiles"   # root folder for Experiment 1
PRODUCT_PATH  = "./Data/Scalability/Products"   # root folder for Experiment 2
SAVE_PATH     = "./Results/Scalability"         # output folder

ITER_PROFILES = 19       # sweep folders 0 ... 19
ITER_PRODUCTS = 2        # sweep folders 0 ... 2

C            = 4         # assortment capacity
MIP_TIME_LIM = 1800      # Gurobi time limit per solve (seconds)
MIP_GAP      = 0.025     # Gurobi optimality gap (used for product-scaling experiment only)
```

Note: `MIP_GAP` is applied only in Experiment 2 (product scaling). Experiment 1 (profile scaling) runs with the default Gurobi gap (`0.0`).

### Running

```bash
julia Run_scalability.jl
```

Both experiments run sequentially in a single invocation. Progress is printed to the terminal for each instance, including the solver name, problem dimensions (N, M), and elapsed time in seconds.

### Output

Results are written to `Results/Scalability/`:

| File | Description |
|---|---|
| `run_times_scale_profiles.xlsx` | running times (seconds) for Greedy-MILP and ColGen across profile-scaling instances |
| `run_times_scale_products.xlsx` | running times (seconds) for Greedy-MILP and ColGen across product-scaling instances |

Each file contains one row per instance with three columns: `instance` (folder index), `Greedy_MILP` (elapsed seconds), and `ColGen` (elapsed seconds).

---

## Reproducibility

The simulation random seed is fixed at `Random.seed!(7)` in Section 6 of `Main.jl`.
