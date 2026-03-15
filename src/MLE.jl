###############################################################################################
#  MLE.jl — log-likelihood, and gradient functions. 
#  
#  The functions loglike and grad implement the MNL maximum-likelihood estimation (MLE) used
#  by all three policies. They rely on two module-level variables that are set
#  immediately before each call to Optim.optimize:
#
#    data   :: Matrix  — rows: [purchase indicators (N+1 cols) | profile index]
#    assort :: Matrix  — rows: offered assortment indicator vectors (N cols)
#
#  Both are declared in the policy runner before calling the optimiser.

# The function loglike_TS implements the bootstrap approximation of Thompson sampling policy. 
# See Section EC.2 in the electronic companion of the paper for details. 
###############################################################################################

"""
    loglike(Beta) -> Float64

Negative MNL log-likelihood for coefficient matrix `Beta` (N × D).

The function returns the *negative* log-likelihood for minimization.
"""
function loglike(Beta::Matrix{Float64}, inst::InstanceData,
                 data::Matrix, assort::Matrix)::Float64
    XX = inst.X'    # (M × D) — one row per profile
    N  = inst.N;  M = inst.M

    ll = 0.0
    for m in 1:M
        idx = findall(==(m), data[:, N+2])
        isempty(idx) && continue

        Sample_m = data[idx, 1:N+1]
        S_m      = assort[idx, :]
        T_m      = size(S_m, 1)
        x_m      = XX[m, :]
        v_m      = exp.(Beta * x_m)          # N-vector of product attractiveness

        # MNL Choice probabilities
        p_m = [S_m .* repeat(v_m', T_m, 1)  ones(T_m, 1)] ./
              repeat(1.0 .+ S_m * v_m, 1, N+1)

        ll += sum(log.(max.(sum(Sample_m .* p_m; dims=2), 1e-300)))
    end

    return -ll   # return negative for minimization
end

"""
    grad(Beta, inst, data, assort) -> Matrix{Float64}

Gradient of the negative MNL log-likelihood with respect to `Beta` (N × D).

Returned matrix has the same shape as `Beta`.
"""
function grad(Beta::Matrix{Float64}, inst::InstanceData,
              data::Matrix, assort::Matrix)::Matrix{Float64}
    XX = inst.X'
    N  = inst.N;  D = inst.D;  M = inst.M

    g_total = zeros(N, D)
    for m in 1:M
        idx = findall(==(m), data[:, N+2])
        isempty(idx) && continue

        Sample_m = data[idx, 1:N+1]
        S_m      = assort[idx, :]
        T_m      = size(S_m, 1)
        x_m      = XX[m, :]
        v_m      = exp.(Beta * x_m)

        # Score for each observation: (purchase indicator) − (choice probability)
        score_m = Sample_m[:, 1:N] .-
                  (S_m .* repeat(v_m', T_m, 1) ./
                   repeat(1.0 .+ S_m * v_m, 1, N))   # (T_m × N)

        # Accumulate gradient
        for t in 1:T_m
            g_total .+= score_m[t, :] * x_m'
        end
    end

    return -g_total 
end

"""
    loglike_TS(Beta, inst, data, assort, Beta_prior, sigma) -> Float64

Penalized negative log-likelihood for Thompson Sampling.

Adds a Mahalanobis-distance regularization term to the MNL log-likelihood:

    objective = −log L(Beta) + (Beta − Beta_prior)ᵀ Σ⁻¹ (Beta − Beta_prior),

where Beta_prior is the sampled prior draw and Sigma is the prior covariance.


See Section EC.2 in the electronic companion of the paper for details on the
    bootstrap approximation of Thompson Sampling.

"""
function loglike_TS(Beta::Matrix{Float64}, inst::InstanceData,
                    data::Matrix, assort::Matrix,
                    Beta_prior::Matrix{Float64},
                    sigma::Matrix{Float64})::Float64
    ll   = -loglike(Beta, inst, data, assort)   # positive log-likelihood
    diff = reshape(Beta .- Beta_prior, 1, inst.N * inst.D)
    pen  = (diff * sigma * diff')[1]
    return -(ll - pen)                          
end
