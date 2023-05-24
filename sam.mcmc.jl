####
#### An MCMC Algorithm for the Non-preferential Model (SAM)
####

function samMCMC(Y, T_ind, X, H, W, nMCMC, nburn)

    ### Y is TxJ matrix containing the counts of J species throughout the study period
    ### T_ind is a vector of length T that determines if counts were observed for each day
    ### X is the Txp covariate matrix with an intercept column
    ### H is the TxJ matrix of lengths of trapping occassions
    ### W is the TxJ matrix of proportion of traps sample
    ### nMCMC is the number of simulations we will run the algorithm
    ### nburn is our burn-in period
    
    T = size(Y)[1]
    J = size(Y)[2]
    p = size(X)[2]

    ###
    ### Save Matrices
    ###

    lam_save = zeros(T, J, nMCMC-nburn)
    gt_save = zeros(T, J, nMCMC-nburn)
    beta_save = zeros(p, J, nMCMC-nburn)
    alpha_save = zeros(J, nMCMC-nburn)
    s2_save = zeros(nMCMC-nburn)

    ###
    ### Hyperparameters
    ###

    ## tuning parameter for llam random-walk
    llam_tune = 1

    ## priors for mu_beta 
    mu_0 = repeat([0.1], p)
    Sig_0_inv = 0*Matrix(I, p, p)

    ## priors for Sigma_beta
    nu = p
    psi = Matrix(I, p, p)

    ## priors for alpha
    mu_alpha = 0
    s2_alpha = 10

    ## priors for s2
    q = 0.001
    r = 1000

    ## priors for llamj1
    mu_1 = 0
    s2_1 = 2

    ###
    ### Initialize Values
    ###

    ## generate initial values
    beta = zeros(p, J)
    alpha = ones(J)
    llamj1 = 5*ones(J)
    mu_beta = zeros(p)
    s2 = 5
    Sig_beta = Matrix(I, p, p)
    Sig_beta_inv = inv(Sig_beta)
    lam = 5*ones(T, J)
    llam = log.(lam)

    ## store initial values in save matrices
    beta_save[:, :, 1] = beta
    alpha_save[:, 1] = alpha
    s2_save[1] = s2
    lam_save[:, :, 1] = lam
    gt_save[:, :, 1] = zeros(T, J)

    ###
    ### Gibbs Sampler
    ###

    for k in ProgressBar(2:nMCMC)

        ###
        ### Update λₜ (t = 2, ..., T-1) for each species
        ###

        for t in 2:(T-1)
            for j in 1:J

                ## random-walk proposal
                llam_star = rand(Normal(llam[t, j], sqrt(llam_tune)), 1)[1]
                lam_star = exp(llam_star)

                ## M-H numerator
                mh1 = logpdf(Normal(transpose(X[t+1, :] - alpha[j]*X[t, :])*beta[:, j] + alpha[j]*llam_star, sqrt(s2)), llam[t+1, j]) + logpdf(Normal(transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j] + alpha[j]*llam[t-1, j], sqrt(s2)), llam_star)

                ## M-H denominator
                mh2 = logpdf(Normal(transpose(X[t+1, :] - alpha[j]*X[t, :])*beta[:, j] + alpha[j]*llam[t, j], sqrt(s2)), llam[t+1, j]) + logpdf(Normal(transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j] + alpha[j]*llam[t-1, j], sqrt(s2)), llam[t, j])

                ## if data was observed
                if T_ind[t] == 1
                    mh1 += logpdf(Poisson(lam_star*H[t, j]*W[t, j]), Y[t, j])
                    mh2 += logpdf(Poisson(lam[t, j]*H[t, j]*W[t, j]), Y[t, j])
                end

                ## acceptance criteria
                mh = exp(mh1 - mh2)
                if mh > rand(1)[1]
                    llam[t, j] = llam_star
                    lam[t, j] = lam_star
                end
            end
        end

        ###
        ### Update λₜ (t = T) for each species
        ###

        for j in 1:J
            
            ## random-walk proposal
            llam_star = rand(Normal(llam[T, j], sqrt(llam_tune)), 1)[1]
            lam_star = exp(llam_star)

            ## M-H numerator
            mh1 = logpdf(Normal(transpose(X[T, :] - alpha[j]*X[T-1, :])*beta[:, j] + alpha[j]*llam[T-1, j], sqrt(s2)), llam_star)

            ## M-H denominator
            mh2 = logpdf(Normal(transpose(X[T, :] - alpha[j]*X[T-1, :])*beta[:, j] + alpha[j]*llam[T-1, j], sqrt(s2)), llam[T, j])

            ## if data were observed
            if T_ind[T] == 1
                mh1 += logpdf(Poisson(lam_star*H[T, j]*W[T, j]), Y[T, j])
                mh2 += logpdf(Poisson(lam[T, j]*H[T, j]*W[T, j]), Y[T, j])
            end

            ## acceptance criteria
            mh = exp(mh1 - mh2)
            if mh > rand(1)[1]
                llam[T, j] = llam_star
                lam[T, j] = lam_star
            end
        end

        ###
        ### Update βs
        ###

        for j in 1:J
            tmpSum1 = zeros(p, p)
            tmpSum2 = zeros(p)
            for t in 2:T
                tmpSum1 += (X[t, :] - alpha[j]*X[t-1, :])*transpose(X[t, :] - alpha[j]*X[t-1, :])
                tmpSum2 += (llam[t, j] - alpha[j]*llam[t-1, j])*(X[t, :] - alpha[j]*X[t-1, :])
            end
            tmpVar = inv(tmpSum1/s2 + Sig_beta_inv)
            tmpMn = vec(tmpVar*(Sig_beta_inv*mu_beta + (tmpSum2)/s2))
            beta[:, j] = rand(MvNormal(tmpMn, Hermitian(tmpVar)), 1)
        end

        ###
        ### Update μ_β
        ###

        tmpVar = inv(Sig_0_inv + J*Sig_beta_inv)
        tmpSum = zeros(p)
        for j in 1:J
            tmpSum += Sig_beta_inv*beta[:, j]
        end
        tmpMn = tmpVar*(Sig_0_inv*mu_0 + tmpSum)
        mu_beta = rand(MvNormal(tmpMn, Hermitian(tmpVar)), 1)

        ###
        ### Update Σ_β
        ###

        tmpNu = J + nu
        tmpSum = zeros(p, p)
        for j in 1:J
            tmpSum += (beta[:, j] - mu_beta)*transpose(beta[:, j] - mu_beta)
        end
        tmpPsi = psi + tmpSum
        Sig_beta = rand(InverseWishart(tmpNu, tmpPsi), 1)[1]
        Sig_beta_inv = inv(Sig_beta)

        ###
        ### Update αs
        ###

        for j in 1:J
            tmpSum1 = 0
            tmpSum2 = 0
            for t in 2:T
                tmpSum1 += (llam[t-1, j] - transpose(X[t-1, :])*beta[:, j])^2
                tmpSum2 += (llam[t, j] - transpose(X[t, :])*beta[:, j])*(llam[t-1, j] - transpose(X[t-1, :])*beta[:, j])
            end
            tmpVar = inv(tmpSum1/s2 + 1/s2_alpha)
            tmpMn = tmpVar*(tmpSum2/s2 + mu_alpha/s2_alpha)
            alpha[j] = rand(Normal(tmpMn, sqrt(tmpVar)), 1)[1]
        end

        ###
        ### Update σ² 
        ###

        tmpSum = 0
        for t in 2:T
            for j in 1:J 
                tmpSum += (llam[t, j] - transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j] - alpha[j]*llam[t-1, j])^2
            end
        end
        tmpq = (T-1)*J/2 + q
        tmpr = inv(tmpSum/2 + 1/r)
        s2 = 1/rand(Gamma(tmpq, tmpr), 1)[1]

        ###
        ### Update log(λ)_1
        ###

        for j in 1:J
            tmpVar = 1/(alpha[j]^2/s2 + 1/s2_1)
            tmpMn = tmpVar*((llam[2, j]*alpha[j] - alpha[j]*transpose(X[2, :] - alpha[j]*X[1, :])*beta[:, j])/s2 + mu_1/s2_1)
            llamj1[j] = rand(Normal(tmpMn, sqrt(tmpVar)), 1)[1]
            llam[1, j] = llamj1[j]
            lam[1, j] = exp(llamj1[j])
        end

        ###
        ### Save Output After Burn-in
        ###

        if k > nburn
            beta_save[:, :, k-nburn] = beta
            alpha_save[:, k-nburn] = alpha
            s2_save[k-nburn] = s2
            lam_save[:, :, k-nburn] = lam

            for j in 1:J
                tmprt = exp(transpose(X[1, :] - alpha[j]*X[1, :])*beta[:, j])
                gt_save[1, j, k-nburn] = (tmprt)*(lam[1, j])^(alpha[j] - 1) - 1
                for t in 2:T
                    tmprt = exp(transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j])
                    gt_save[t, j, k-nburn] = (tmprt)*(lam[t-1, j])^(alpha[j] - 1) - 1
                end
            end
        end
    end

    ###
    ### Write Output
    ###

    return [alpha_save, beta_save, lam_save, s2_save, gt_save]

end