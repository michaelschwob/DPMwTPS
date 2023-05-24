###
### Script for Simulated Case Study
###

###
### Set-up
###

using Random, CSV, DataFrames, Statistics, Query, StatsBase, LinearAlgebra, Distributions, LogExpFunctions
using Makie, CairoMakie, PDFmerger # for plotting
using Dates # for time comparison
using RCall # for Ridge Script
import PlotlyJS # for boxplots
using ProgressBars # for determining progress
using JLD2 # environments

println("Time started: ", Dates.Time(Dates.now()), ".")

## increase axis label size
fontsize_theme = Theme(fontsize = 30)
set_theme!(fontsize_theme)

## set seed
Random.seed!(702)
nmcmc = 500000

## Get MCMC Functions
include("sam.mcmc.jl")
include("spam.mcmc.probit.jl")

###
### Read Weather Data for Covariates
###

weather_data = CSV.read("Weather_iter.csv", DataFrame, missingstring = "NA")
GDD = weather_data |> @filter(_.Site == "HARV") |> @select(:GDD) |> @dropna() |> DataFrame |> Matrix
nyears = 3
T = 365*nyears # number of days in study period

###
### Moving Average Function to Smooth GDD
###

function moving_average(theta_basis, covariate)
    result = zeros(length(covariate))
    for i in 1:length(covariate)
        if i <= theta_basis
            result[i] = mean(skipmissing(covariate[1:i]))
        else 
            result[i] = mean(skipmissing(covariate[(i-theta_basis+1):i]))  
        end
    end
    return result
end

X = [ones(T) moving_average(14, GDD)[1:T]]

years = [1 2 3]
yearsAug = minimum(years):(maximum(years)+1)
vdays = [(yearsAug[i] - minimum(years))*365 for i in 1:length(yearsAug)]
midpoints = vdays[2:end] - 365/2*ones(nyears)

###
### Simulate True Abundance
###

## true parameters
llam = zeros(T)
p = 2 # number of covariates including intercept
alpha = 0.98
beta = [0.1, 0.3]
s2 = 0.03

## simulate log-abundance process
llam[1] = log(2)
for t in 2:T
    llam[t] = rand(Normal((X[t, :]' - alpha*X[t-1, :]')*beta + alpha*llam[t-1], sqrt(s2)), 1)[1]
end
lam = exp.(llam)

###
### Plot Simulated Abundance
###

truePlot = Figure(resolution = (1300, 900))
gl = truePlot[1, 1] = GridLayout()

axalpha = Axis(gl[1, 1], xlabel = "Day of Study", ylabel = "Abundance", title = "Simulated Abundance")
lines!(1:T, lam, color = ("black", 0.5))
save("trueAbundance.png", truePlot)

###
### Obtain Observations Under Different Sampling Mechanisms
###

## Scenario 1: Complete Sampling
## We don't report this in our paper. This is just to test that the model works correctly
## under complete observation - which it does :)
T1 = ones(T)

## Scenario 2: Random Sampling
T2 = rand(Bernoulli(0.3), T)

## Scenario 3: Binary Sampling
T3 = zeros(T)
for t in 1:T
    if lam[t] > 15
        pt = 0.3
    else 
        pt = 0
    end
    T3[t] = rand(Bernoulli(pt), 1)[1]
end

## Scenario 4: Logistic Sampling
T4 = zeros(T)
for t in 1:T
    tmpx = -10 + 0.4*lam[t] 
    pt = 1/(1+exp(-tmpx))
    T4[t] = rand(Bernoulli(pt), 1)[1]
end

## Actual Observations
Y = zeros(T)
for t in 1:T
    Y[t] = rand(Poisson(lam[t]), 1)[1]
end

@save "env.jld2" T1 T2 T3 T4 Y T lam

###
### Plot Observed Abundances
###

obsPlot = Figure(resolution = (1300, 900))
gl = obsPlot[1, 1] = GridLayout()

## Scenario 2
axalpha = Axis(gl[1, 1], xlabel = "Day of Study", ylabel = "Abundance", title = "(a) Random Sampling")
lines!(1:T, lam, color = ("black", 0.5))
scatter!((1:T)[findall(T2.==1)], Y[findall(T2.==1)], color = ("red", 0.5), markersize = 5)

## Scenario 3
axalpha = Axis(gl[2, 1], xlabel = "Day of Study", ylabel = "Abundance", title = "(b) Preferential Switch Sampling")
lines!(1:T, lam, color = ("black", 0.5))
scatter!((1:T)[findall(T3.==1)], Y[findall(T3.==1)], color = ("red", 0.5), markersize = 5)

## Scenario 4
axalpha = Axis(gl[3, 1], xlabel = "Day of Study", ylabel = "Abundance", title = "(c) Logistic Sampling")
lines!(1:T, lam, color = ("black", 0.5))
scatter!((1:T)[findall(T4.==1)], Y[findall(T4.==1)], color = ("red", 0.5), markersize = 5)
save("obsPlot.png", obsPlot)

###
### Prepare Model Fitting and Plotting
###

nburn = round(Int64, 0.8*nmcmc)
H = ones(T, 1)
W = ones(T, 1)

estPlot = Figure(resolution = (2000, 1750), fontsize = 30)
gl = estPlot[1, 1] = GridLayout()

## Labels for Box Plots
labels = ["Random" for i in 1:(nmcmc-nburn)]
append!(labels, ["Preferential Switch" for i in 1:(nmcmc-nburn)])
append!(labels, ["Logistic" for i in 1:(nmcmc-nburn)])

###
### Root Mean Squared Error (RMSE) Function
###

function RMSE(truth, pred)
    tmpSum = 0
    tmpT = length(truth)
    for t in 1:tmpT
        tmpSum += (truth[t] - pred[t])^2
    end
    return sqrt(tmpSum/tmpT)
end

RMSEMat = zeros(4, 2) # for each scenario, we have two competing models

####
#### Scenario 2
####

println("\nScenario 2.")

sam_out = samMCMC(Y[:, :], T2, X, H, W, nmcmc, nburn)
println("Finished SAM.")

spam_out = spamThreshMCMC(Y[:, :], T2, X, H, W, nmcmc, nburn)
println("Finished SPAM.")

samDict = Dict("alpha" => sam_out[1], "beta" => sam_out[2], "lam" => sam_out[3], "s2" => sam_out[4], "gt" => sam_out[5])
sam_out = nothing

spamDict = Dict("alpha" => spam_out[1], "beta" => spam_out[2], "lam" => spam_out[3], "s2" => spam_out[4], "gt" => spam_out[5], "theta0" => spam_out[6], "theta1" => spam_out[7], "lamThresh" => spam_out[8])
spam_out  =  nothing

println("Wrote dictionaries.")
@save "scen2.jld2" samDict spamDict

###
### Obtain Posterior Abundance Estimates
###

lamPointSam = zeros(T)
lamLBSam = zeros(T)
lamUBSam = zeros(T)
lamPointSpam = zeros(T)
lamLBSpam = zeros(T)
lamUBSpam = zeros(T)

for t in 1:T
    lamPointSam[t] = mean(samDict["lam"][t, 1, :])
    lamLBSam[t] = quantile(samDict["lam"][t, 1, :], 0.025)
    lamUBSam[t] = quantile(samDict["lam"][t, 1, :], 0.975)
    lamPointSpam[t] = mean(spamDict["lam"][t, 1, :])
    lamLBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.025)
    lamUBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.975)
end

## for boxPlot outputs
alphaSam = samDict["alpha"][1, :]
alphaSpam = spamDict["alpha"][1, :]
beta1Sam = samDict["beta"][1, 1, :]
beta1Spam = spamDict["beta"][1, 1, :]
beta2Sam = samDict["beta"][2, 1, :]
beta2Spam = spamDict["beta"][2, 1, :]
s2Sam = samDict["s2"][:]
s2Spam = spamDict["s2"][:]

## for new boxPlot plot
alphaSam2 = samDict["alpha"][1, :]
alphaSpam2 = spamDict["alpha"][1, :]
beta1Sam2 = samDict["beta"][1, 1, :]
beta1Spam2 = spamDict["beta"][1, 1, :]
beta2Sam2 = samDict["beta"][2, 1, :]
beta2Spam2 = spamDict["beta"][2, 1, :]
s2Sam2 = samDict["s2"][:]
s2Spam2 = spamDict["s2"][:]
@rput alphaSam2 alphaSpam2 beta1Sam2 beta1Spam2 beta2Sam2 beta2Spam2 s2Sam2 s2Spam2
R"""
alphaSam2 = alphaSam2[!alphaSam2 %in% boxplot.stats(alphaSam2)$out]
alphaSpam2 = alphaSpam2[!alphaSpam2 %in% boxplot.stats(alphaSpam2)$out]
beta1Sam2 = beta1Sam2[!beta1Sam2 %in% boxplot.stats(beta1Sam2)$out]
beta1Spam2 = beta1Spam2[!beta1Spam2 %in% boxplot.stats(beta1Spam2)$out]
beta2Sam2 = beta2Sam2[!beta2Sam2 %in% boxplot.stats(beta2Sam2)$out]
beta2Spam2 = beta2Spam2[!beta2Spam2 %in% boxplot.stats(beta2Spam2)$out]
s2Sam2 = s2Sam2[!s2Sam2 %in% boxplot.stats(s2Sam2)$out]
s2Spam2 = s2Spam2[!s2Spam2 %in% boxplot.stats(s2Spam2)$out]
"""
@rget alphaSam2 alphaSpam2 beta1Sam2 beta1Spam2 beta2Sam2 beta2Spam2 s2Sam2 s2Spam2

## for testing preferential sampling
println("Lamda tilde post mean: ", mean(spamDict["lamThresh"]))
println("Theta1 post mean: ", mean(spamDict["theta1"]))
println("Theta1 LB: ", quantile(spamDict["theta1"], 0.025))
println("Theta1 UB: ", quantile(spamDict["theta1"], 0.975))

samDict = nothing
spamDict = nothing

###
### Plot Competing Inference
###

axest = Axis(gl[1, 1], title = "(a) Random Sampling", xlabel = "Days", ylabel = "Abundance", xgridvisible = false, ygridvisible = false)
scatter!((1:T)[findall(T2 .== 1)], Y[findall(T2 .== 1)], color = :black, markersize = 11, label = "Observations")
band!(1:T, lamLBSam[:, 1], lamUBSam[:, 1], color = ("red", 0.25))
band!(1:T, lamLBSpam[:, 1], lamUBSpam[:, 1], color = ("blue", 0.25))
lines!(1:T, lam, color = ("black", 0.5), label = "Truth", linewidth = 4)
lines!(1:T, lamPointSam[:, 1], color = ("red", 1), label = "Non-preferential", linewidth = 4, linestyle = :dot)
lines!(1:T, lamPointSpam[:, 1], color = ("blue", 1), label = "Preferential", linewidth = 4, linestyle = :dashdot)
vlines!(vdays, color = (:black, 0.5))
axislegend("Legend")

RMSEMat[2, 1] = RMSE(lam[findall(T2 .== 0)], lamPointSam[findall(T2 .== 0)])
RMSEMat[2, 2] = RMSE(lam[findall(T2 .== 0)], lamPointSpam[findall(T2 .== 0)])

####
#### Scenario 3
####

println("\nScenario 3.")

sam_out = samMCMC(Y[:, :], T3, X, H, W, nmcmc, nburn)
println("Finished SAM.")

spam_out = spamThreshMCMC(Y[:, :], T3, X, H, W, nmcmc, nburn)
println("Finished SPAM.")

samDict = Dict("alpha" => sam_out[1], "beta" => sam_out[2], "lam" => sam_out[3], "s2" => sam_out[4], "gt" => sam_out[5])
sam_out = nothing

spamDict = Dict("alpha" => spam_out[1], "beta" => spam_out[2], "lam" => spam_out[3], "s2" => spam_out[4], "gt" => spam_out[5], "theta0" => spam_out[6], "theta1" => spam_out[7], "lamThresh" => spam_out[8])
spam_out  =  nothing

println("Wrote dictionaries.")
@save "scen3.jld2" samDict spamDict

###
### Obtain Posterior Abundance Estimates
###

lamPointSam = zeros(T)
lamLBSam = zeros(T)
lamUBSam = zeros(T)
lamPointSpam = zeros(T)
lamLBSpam = zeros(T)
lamUBSpam = zeros(T)

for t in 1:T
    lamPointSam[t] = mean(samDict["lam"][t, 1, :])
    lamLBSam[t] = quantile(samDict["lam"][t, 1, :], 0.025)
    lamUBSam[t] = quantile(samDict["lam"][t, 1, :], 0.975)
    lamPointSpam[t] = mean(spamDict["lam"][t, 1, :])
    lamLBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.025)
    lamUBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.975)
end

## add boxplot outputs
append!(alphaSam, samDict["alpha"][1, :])
append!(alphaSpam, spamDict["alpha"][1, :])
append!(beta1Sam, samDict["beta"][1, 1, :])
append!(beta1Spam, spamDict["beta"][1, 1, :])
append!(beta2Sam, samDict["beta"][2, 1, :])
append!(beta2Spam, spamDict["beta"][2, 1, :])
append!(s2Sam, samDict["s2"][:])
append!(s2Spam, spamDict["s2"][:])

## for new boxPlot plot
alphaSam3 = samDict["alpha"][1, :]
alphaSpam3 = spamDict["alpha"][1, :]
beta1Sam3 = samDict["beta"][1, 1, :]
beta1Spam3 = spamDict["beta"][1, 1, :]
beta2Sam3 = samDict["beta"][2, 1, :]
beta2Spam3 = spamDict["beta"][2, 1, :]
s2Sam3 = samDict["s2"][:]
s2Spam3 = spamDict["s2"][:]
@rput alphaSam3 alphaSpam3 beta1Sam3 beta1Spam3 beta2Sam3 beta2Spam3 s2Sam3 s2Spam3
R"""
alphaSam3 = alphaSam3[!alphaSam3 %in% boxplot.stats(alphaSam3)$out]
alphaSpam3 = alphaSpam3[!alphaSpam3 %in% boxplot.stats(alphaSpam3)$out]
beta1Sam3 = beta1Sam3[!beta1Sam3 %in% boxplot.stats(beta1Sam3)$out]
beta1Spam3 = beta1Spam3[!beta1Spam3 %in% boxplot.stats(beta1Spam3)$out]
beta2Sam3 = beta2Sam3[!beta2Sam3 %in% boxplot.stats(beta2Sam3)$out]
beta2Spam3 = beta2Spam3[!beta2Spam3 %in% boxplot.stats(beta2Spam3)$out]
s2Sam3 = s2Sam3[!s2Sam3 %in% boxplot.stats(s2Sam3)$out]
s2Spam3 = s2Spam3[!s2Spam3 %in% boxplot.stats(s2Spam3)$out]
"""
@rget alphaSam3 alphaSpam3 beta1Sam3 beta1Spam3 beta2Sam3 beta2Spam3 s2Sam3 s2Spam3

samDict = nothing
spamDict = nothing

###
### Plot Competing Inference
###

axest = Axis(gl[2, 1], title = "(b) Preferential Switch Sampling", xlabel = "Days", ylabel = "Abundance", xgridvisible = false, ygridvisible = false)
scatter!((1:T)[findall(T3 .== 1)], Y[findall(T3 .== 1)], color = :black, markersize = 12, label = "Observations")
band!(1:T, lamLBSam[:, 1], lamUBSam[:, 1], color = ("red", 0.25))
band!(1:T, lamLBSpam[:, 1], lamUBSpam[:, 1], color = ("blue", 0.25))
lines!(1:T, lam, color = ("black", 0.5), label = "Truth", linewidth = 4)
lines!(1:T, lamPointSam[:, 1], color = ("red", 1), label = "Non-preferential", linewidth = 4, linestyle = :dot)
lines!(1:T, lamPointSpam[:, 1], color = ("blue", 1), label = "Preferential", linewidth = 4, linestyle = :dashdot)
vlines!(vdays, color = (:black, 0.5))
axislegend("Legend")

RMSEMat[3, 1] = RMSE(lam[findall(T3 .== 0)], lamPointSam[findall(T3 .== 0)])
RMSEMat[3, 2] = RMSE(lam[findall(T3 .== 0)], lamPointSpam[findall(T3 .== 0)])

####
#### Scenario 4
####

println("\nScenario 4.")

sam_out = samMCMC(Y[:, :], T4, X, H, W, nmcmc, nburn)
println("Finished SAM.")

spam_out = spamThreshMCMC(Y[:, :], T4, X, H, W, nmcmc, nburn)
println("Finished SPAM.")

samDict = Dict("alpha" => sam_out[1], "beta" => sam_out[2], "lam" => sam_out[3], "s2" => sam_out[4], "gt" => sam_out[5])
sam_out = nothing

spamDict = Dict("alpha" => spam_out[1], "beta" => spam_out[2], "lam" => spam_out[3], "s2" => spam_out[4], "gt" => spam_out[5], "theta0" => spam_out[6], "theta1" => spam_out[7], "lamThresh" => spam_out[8])
spam_out  =  nothing

println("Wrote dictionaries.")
@save "scen4.jld2" samDict spamDict

###
### Obtain Posterior Abundance Estimates
###

lamPointSam = zeros(T)
lamLBSam = zeros(T)
lamUBSam = zeros(T)
lamPointSpam = zeros(T)
lamLBSpam = zeros(T)
lamUBSpam = zeros(T)

for t in 1:T
    lamPointSam[t] = mean(samDict["lam"][t, 1, :])
    lamLBSam[t] = quantile(samDict["lam"][t, 1, :], 0.025)
    lamUBSam[t] = quantile(samDict["lam"][t, 1, :], 0.975)
    lamPointSpam[t] = mean(spamDict["lam"][t, 1, :])
    lamLBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.025)
    lamUBSpam[t] = quantile(spamDict["lam"][t, 1, :], 0.975)
end

## add boxplot outputs
append!(alphaSam, samDict["alpha"][1, :])
append!(alphaSpam, spamDict["alpha"][1, :])
append!(beta1Sam, samDict["beta"][1, 1, :])
append!(beta1Spam, spamDict["beta"][1, 1, :])
append!(beta2Sam, samDict["beta"][2, 1, :])
append!(beta2Spam, spamDict["beta"][2, 1, :])
append!(s2Sam, samDict["s2"][:])
append!(s2Spam, spamDict["s2"][:])

## for new boxPlot plot
alphaSam4 = samDict["alpha"][1, :]
alphaSpam4 = spamDict["alpha"][1, :]
beta1Sam4 = samDict["beta"][1, 1, :]
beta1Spam4 = spamDict["beta"][1, 1, :]
beta2Sam4 = samDict["beta"][2, 1, :]
beta2Spam4 = spamDict["beta"][2, 1, :]
s2Sam4 = samDict["s2"][:]
s2Spam4 = spamDict["s2"][:]
@rput alphaSam4 alphaSpam4 beta1Sam4 beta1Spam4 beta2Sam4 beta2Spam4 s2Sam4 s2Spam4
R"""
alphaSam4 = alphaSam4[!alphaSam4 %in% boxplot.stats(alphaSam4)$out]
alphaSpam4 = alphaSpam4[!alphaSpam4 %in% boxplot.stats(alphaSpam4)$out]
beta1Sam4 = beta1Sam4[!beta1Sam4 %in% boxplot.stats(beta1Sam4)$out]
beta1Spam4 = beta1Spam4[!beta1Spam4 %in% boxplot.stats(beta1Spam4)$out]
beta2Sam4 = beta2Sam4[!beta2Sam4 %in% boxplot.stats(beta2Sam4)$out]
beta2Spam4 = beta2Spam4[!beta2Spam4 %in% boxplot.stats(beta2Spam4)$out]
s2Sam4 = s2Sam4[!s2Sam4 %in% boxplot.stats(s2Sam4)$out]
s2Spam4 = s2Spam4[!s2Spam4 %in% boxplot.stats(s2Spam4)$out]
"""
@rget alphaSam4 alphaSpam4 beta1Sam4 beta1Spam4 beta2Sam4 beta2Spam4 s2Sam4 s2Spam4

samDict = nothing
spamDict = nothing

###
### Plot Competing Inference
###

axest = Axis(gl[3, 1], title = "(c) Logistic Sampling", xlabel = "Days", ylabel = "Abundance", xgridvisible = false, ygridvisible = false)
scatter!((1:T)[findall(T4 .== 1)], Y[findall(T4 .== 1)], color = :black, markersize = 12, label = "Observations")
band!(1:T, lamLBSam[:, 1], lamUBSam[:, 1], color = ("red", 0.25))
band!(1:T, lamLBSpam[:, 1], lamUBSpam[:, 1], color = ("blue", 0.25))
lines!(1:T, lam, color = ("black", 0.5), label = "Truth", linewidth = 4)
lines!(1:T, lamPointSam[:, 1], color = ("red", 1), label = "Non-preferential", linewidth = 4, linestyle = :dot)
lines!(1:T, lamPointSpam[:, 1], color = ("blue", 1), label = "Preferential", linewidth = 4, linestyle = :dashdot)
vlines!(vdays, color = (:black, 0.5))
axislegend("Legend")

RMSEMat[4, 1] = RMSE(lam[findall(T4 .== 0)], lamPointSam[findall(T4 .== 0)])
RMSEMat[4, 2] = RMSE(lam[findall(T4 .== 0)], lamPointSpam[findall(T4 .== 0)])

###
### Box Plots
###

## combine output into one vector
alphaSam = [vec(alphaSam2); vec(alphaSam3); vec(alphaSam4)]
alphaSpam = [vec(alphaSpam2); vec(alphaSpam3); vec(alphaSpam4)]
beta1Sam = [vec(beta1Sam2); vec(beta1Sam3); vec(beta1Sam4)]
beta1Spam = [vec(beta1Spam2); vec(beta1Spam3); vec(beta1Spam4)]
beta2Sam = [vec(beta2Sam2); vec(beta2Sam3); vec(beta2Sam4)]
beta2Spam = [vec(beta2Spam2); vec(beta2Spam3); vec(beta2Spam4)]
s2Sam = [vec(s2Sam2); vec(s2Sam3); vec(s2Sam4)]
s2Spam = [vec(s2Spam2); vec(s2Spam3); vec(s2Spam4)]

## get labels for boxplots
alphaLabelSam = ["Random" for i in 1:length(alphaSam2)]
append!(alphaLabelSam, ["Preferential Switch" for i in 1:length(alphaSam3)])
append!(alphaLabelSam, ["Logistic" for i in 1:length(alphaSam4)])
alphaLabelSpam = ["Random" for i in 1:length(alphaSpam2)]
append!(alphaLabelSpam, ["Preferential Switch" for i in 1:length(alphaSpam3)])
append!(alphaLabelSpam, ["Logistic" for i in 1:length(alphaSpam4)])
beta1LabelSam = ["Random" for i in 1:length(beta1Sam2)]
append!(beta1LabelSam, ["Preferential Switch" for i in 1:length(beta1Sam3)])
append!(beta1LabelSam, ["Logistic" for i in 1:length(beta1Sam4)])
beta1LabelSpam = ["Random" for i in 1:length(beta1Spam2)]
append!(beta1LabelSpam, ["Preferential Switch" for i in 1:length(beta1Spam3)])
append!(beta1LabelSpam, ["Logistic" for i in 1:length(beta1Spam4)])
beta2LabelSam = ["Random" for i in 1:length(beta2Sam2)]
append!(beta2LabelSam, ["Preferential Switch" for i in 1:length(beta2Sam3)])
append!(beta2LabelSam, ["Logistic" for i in 1:length(beta2Sam4)])
beta2LabelSpam = ["Random" for i in 1:length(beta2Spam2)]
append!(beta2LabelSpam, ["Preferential Switch" for i in 1:length(beta2Spam3)])
append!(beta2LabelSpam, ["Logistic" for i in 1:length(beta2Spam4)])
s2LabelSam = ["Random" for i in 1:length(s2Sam2)]
append!(s2LabelSam, ["Preferential Switch" for i in 1:length(s2Sam3)])
append!(s2LabelSam, ["Logistic" for i in 1:length(s2Sam4)])
s2LabelSpam = ["Random" for i in 1:length(s2Spam2)]
append!(s2LabelSpam, ["Preferential Switch" for i in 1:length(s2Spam3)])
append!(s2LabelSpam, ["Logistic" for i in 1:length(s2Spam4)])

traceAlphaSam = PlotlyJS.box(y = alphaSam, x = alphaLabelSam, name = "Non-preferential", marker_color = "red", boxpoints = false)
traceAlphaSpam = PlotlyJS.box(y = alphaSpam, x = alphaLabelSpam, name = "Preferential", marker_color = "blue", boxpoints = false)
boxPlotAlpha = PlotlyJS.plot([traceAlphaSam, traceAlphaSpam], PlotlyJS.Layout(yaxis_title = "α", boxmode = "group", legend = PlotlyJS.attr(orientation = "h", yanchor = "bottom", xanchor = "right", x = 1, y = 1.02), size = (2000, 500), font_size = 24))
PlotlyJS.add_hline!(boxPlotAlpha, alpha)

traceBeta1Sam = PlotlyJS.box(y = beta1Sam, x = beta1LabelSam, name = "Non-preferential", marker_color = "red", boxpoints = false)
traceBeta1Spam = PlotlyJS.box(y = beta1Spam, x = beta1LabelSpam, name = "Preferential", marker_color = "blue", boxpoints = false)
boxPlotBeta1 = PlotlyJS.plot([traceBeta1Sam, traceBeta1Spam], PlotlyJS.Layout(yaxis_title = "β₁", boxmode = "group", legend = PlotlyJS.attr(orientation = "h", yanchor = "bottom", xanchor = "right", x = 1, y = 1.02), size = (2000, 500), font_size = 24))
PlotlyJS.add_hline!(boxPlotBeta1, beta[1])

traceBeta2Sam = PlotlyJS.box(y = beta2Sam, x = beta2LabelSam, name = "Non-preferential", marker_color = "red", boxpoints = false)
traceBeta2Spam = PlotlyJS.box(y = beta2Spam, x = beta2LabelSpam, name = "Preferential", marker_color = "blue", boxpoints = false)
boxPlotBeta2 = PlotlyJS.plot([traceBeta2Sam, traceBeta2Spam], PlotlyJS.Layout(yaxis_title = "β₂", boxmode = "group", legend = PlotlyJS.attr(orientation = "h", yanchor = "bottom", xanchor = "right", x = 1, y = 1.02), size = (2000, 500), font_size = 24))
PlotlyJS.add_hline!(boxPlotBeta2, beta[2])

traces2Sam = PlotlyJS.box(y = s2Sam, x = s2LabelSam, name = "Non-preferential", marker_color = "red", boxpoints = false)
traces2Spam = PlotlyJS.box(y = s2Spam, x = s2LabelSpam, name = "Preferential", marker_color = "blue", boxpoints = false)
boxPlots2 = PlotlyJS.plot([traces2Sam, traces2Spam], PlotlyJS.Layout(yaxis_title = "σ²", boxmode = "group", legend = PlotlyJS.attr(orientation = "h", yanchor = "bottom", xanchor = "right", x = 1, y = 1.02), size = (2000, 500), font_size = 24))
PlotlyJS.add_hline!(boxPlots2, s2)

boxPlot = [boxPlotAlpha; boxPlots2 ; boxPlotBeta1; boxPlotBeta2]
PlotlyJS.relayout!(boxPlot, title_text = "Posterior Estimates", legend = PlotlyJS.attr(orientation = "h", yanchor = "bottom", xanchor = "right", x = 1, y = 1.02), size = (2000, 500))

###
### Save Plots
###

save("estPlot.png", estPlot)
save("boxAlpha.png", boxPlotAlpha)
save("boxBeta1.png", boxPlotBeta1)
save("boxBeta2.png", boxPlotBeta2)
save("boxs2.png", boxPlots2)

println("Time ended: ", Dates.Time(Dates.now()), ".\n\n")

println(RMSEMat)
CSV.write("RSME.csv", DataFrame(RMSEMat, :auto))