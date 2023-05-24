####
#### An MCMC Algorithm for the Preferential Model (SPAM)
####

function spamMCMC(Y, T_ind, X, H, W, nMCMC, nburn)

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
    theta0_save = zeros(nMCMC-nburn)
    theta1_save = zeros(nMCMC-nburn)
    lamThresh_save = zeros(nMCMC - nburn)

    ####
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

    ## priors for lam_thresh
    alpha_lam = 1
    beta_lam = 20

    ## priors for theta (probit regression)
    Sig_theta_inv = inv(I)
    theta_mn = vec([0, 0])
    tau1 = (T_ind .== 1)
    tau0 = (T_ind .== 0)

    ###
    ### Initialize Values
    ###

    ## generate initial values
    beta = zeros(p, J)
    alpha = ones(J)
    mu_beta = zeros(p)
    s2 = 5
    Sig_beta = Matrix(I, p, p)
    Sig_beta_inv = inv(Sig_beta)
    lam = 5*ones(T, J)
    llam = log.(lam)
    theta = vec([0.5 0.5])
    lam_thresh = 0.5

    ## store initial values in save matrices
    beta_save[:, :, 1] = beta
    alpha_save[:, 1] = alpha
    s2_save[1] = s2
    lam_save[:, :, 1] = lam
    gt_save[:, :, 1] = zeros(T, J)
    theta0_save[1] = theta[1]
    theta1_save[1] = theta[2]
    lamThresh_save[1] = lam_thresh

    ## necessary R things
    @rput X T T_ind tau1 tau0 theta_mn p
    R"""
    library(truncnorm)
    Sig_theta_inv = solve(diag(p))
    """

    ###
    ### Gibbs Sampler
    ###

    for k in ProgressBar(2:nMCMC)

        ###
        ### Update λ_Thresh
        ###

        lamThresh_star = rand(Gamma(alpha_lam, beta_lam), 1)[1]

        ## initialize M-H values
        mh1 = 0
        mh2 = 0

        ## compute contribution of [tau(t_i)|lam_thresh] for each day
        for t in 1:T
            mosPop = sum(lam[t, :]) # calculate day t's population

            probStar = cdf(Normal(0, 1), only(theta'*[1; mosPop >= lamThresh_star]))
            probPast = cdf(Normal(0, 1), only(theta'*[1; mosPop >= lam_thresh]))

            mh1 += logpdf(Bernoulli(probStar), T_ind[t])
            mh2 += logpdf(Bernoulli(probPast), T_ind[t])
        end

        ## acceptance criteria
        mh = exp(mh1 - mh2)
        if mh > rand(1)[1]
            lam_thresh = lamThresh_star
        end

        ###
        ### Update log(λ)_1
        ###

        for j in 1:J 

            ## random-walk proposal
            llam_star = rand(Normal(llam[1, j], sqrt(llam_tune)), 1)[1]
            lam_star = exp(llam_star)

            ## compute p(t_1)
            sumPast = sum(lam[1, :])
            sumStar = sum(lam[1, 1:end .!= j]) + lam_star

            probStar = cdf(Normal(0, 1), only(theta'*[1; sumStar >= lam_thresh]))
            probPast = cdf(Normal(0, 1), only(theta'*[1; sumPast >= lam_thresh]))
            
            ## M-H numerator
            mh1 = logpdf(Bernoulli(probStar), T_ind[1]) + logpdf(Normal(transpose(X[2, :] - alpha[j]*X[1, :])*beta[:, j] + alpha[j]*llam_star, sqrt(s2)), llam[2, j]) + logpdf(Normal(mu_1, sqrt(s2_1)), llam_star)

            ## M-H denominator
            mh2 = logpdf(Bernoulli(probPast), T_ind[1]) + logpdf(Normal(transpose(X[2, :] - alpha[j]*X[1, :])*beta[:, j] + alpha[j]*llam[1, j], sqrt(s2)), llam[2, j]) + logpdf(Normal(mu_1, sqrt(s2_1)), llam[1, j])

            ## if data were observed
            if T_ind[1] == 1
                mh1 += logpdf(Poisson(lam_star*H[1, j]*W[1, j]), Y[1, j])
                mh2 += logpdf(Poisson(lam[1, j]*H[1, j]*W[1, j]), Y[1, j])
            end

            ## acceptance criteria
            mh = exp(mh1 - mh2)
            if mh > rand(1)[1]
                llam[1, j] = llam_star
                lam[1, j] = lam_star
            end
        end

        ###
        ### Update λₜ (t = 2, ..., T-1) for each species
        ###

        for t in 2:(T-1)
            for j in 1:J

                ## random-walk proposal
                llam_star = rand(Normal(llam[t, j], sqrt(llam_tune)), 1)[1]
                lam_star = exp(llam_star)
                
                ## compute p(t_t)
                sumPast = sum(lam[t, :])
                sumStar = sum(lam[t, 1:end .!= j]) + lam_star

                probStar = cdf(Normal(0, 1), only(theta'*[1; sumStar >= lam_thresh]))
                probPast = cdf(Normal(0, 1), only(theta'*[1; sumPast >= lam_thresh]))

                ## M-H numerator
                mh1 = logpdf(Bernoulli(probStar), T_ind[t]) + logpdf(Normal(transpose(X[t+1, :] - alpha[j]*X[t, :])*beta[:, j] + alpha[j]*llam_star, sqrt(s2)), llam[t+1, j]) + logpdf(Normal(transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j] + alpha[j]*llam[t-1, j], sqrt(s2)), llam_star)

                ## M-H denominator
                mh2 = logpdf(Bernoulli(probPast), T_ind[t]) + logpdf(Normal(transpose(X[t+1, :] - alpha[j]*X[t, :])*beta[:, j] + alpha[j]*llam[t, j], sqrt(s2)), llam[t+1, j]) + logpdf(Normal(transpose(X[t, :] - alpha[j]*X[t-1, :])*beta[:, j] + alpha[j]*llam[t-1, j], sqrt(s2)), llam[t, j])

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

            ## compute p(t_T)
            sumPast = sum(lam[T, :])
            sumStar = sum(lam[T, 1:end .!= j]) + lam_star

            probStar = cdf(Normal(0, 1), only(theta'*[1; sumStar >= lam_thresh]))
            probPast = cdf(Normal(0, 1), only(theta'*[1; sumPast >= lam_thresh]))

            ## M-H numerator
            mh1 = logpdf(Bernoulli(probStar), T_ind[T]) + logpdf(Normal(transpose(X[T, :] - alpha[j]*X[T-1, :])*beta[:, j] + alpha[j]*llam[T-1, j], sqrt(s2)), llam_star)

            ## M-H denominator
            mh2 = logpdf(Bernoulli(probPast), T_ind[T]) + logpdf(Normal(transpose(X[T, :] - alpha[j]*X[T-1, :])*beta[:, j] + alpha[j]*llam[T-1, j], sqrt(s2)), llam[T, j])

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
        ### Update Latent Probit Regression Parameters: z
        ###

        @rput theta lam lam_thresh
        R"""
        z = matrix(0, T, 1)
        Xtmp = matrix(1, T, 2)
        for(t in 1:T){
            Xtmp[t, 2] = ifelse(sum(lam[t, ]) >= lam_thresh, 1, 0)
        }
        z1 = rtruncnorm(sum(T_ind), mean = (Xtmp%*%theta)[tau1], sd = 1, a = 0, b = Inf)
        z0 = rtruncnorm(sum(1-T_ind), mean = (Xtmp%*%theta)[tau0], sd = 1, a = -Inf, b = 0)
        z[tau1, ] = z1
        z[tau0, ] = z0
        """
        @rget z

        ###
        ### Update θ
        ###

        @rput z lam lam_thresh
        R"""
        Xtmp = matrix(1, T, 2)
        for(t in 1:T){
            Xtmp[t, 2] = ifelse(sum(lam[t, ]) >= lam_thresh, 1, 0)
        }
        tmpChol = chol(t(Xtmp)%*%Xtmp + Sig_theta_inv)
        theta = backsolve(tmpChol, backsolve(tmpChol, t(Xtmp)%*%z + Sig_theta_inv%*%theta_mn, transpose = TRUE) + rnorm(2))
        """
        @rget theta
        
        ###
        ### Save Output After Burn-in
        ###

        if k > nburn
            beta_save[:, :, k-nburn] = beta
            alpha_save[:, k-nburn] = alpha
            s2_save[k-nburn] = s2
            lam_save[:, :, k-nburn] = lam
            theta0_save[k-nburn] = theta[1]
            theta1_save[k-nburn] = theta[2]
            lamThresh_save[k-nburn] = lam_thresh
    
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

    return [alpha_save, beta_save, lam_save, s2_save, gt_save, theta0_save, theta1_save, lamThresh_save]

end