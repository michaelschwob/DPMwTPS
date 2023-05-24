###
### Script for Plotting Mosquito MCMC Output
###

###
### Set-up
###

using Random, CSV, DataFrames, Statistics, Query, StatsBase, LinearAlgebra, Distributions, LogExpFunctions
using Makie, CairoMakie, PDFmerger # for plotting
using Dates # for time comparison
using RCall # for Ridge Script
using ProgressBars # for determining progress
using JLD2 # environments

## set Directory
pathName = pwd() # get path to all scripts
#cd("Outputw0") # scenario 1
cd("Outputwo0") # scenario 2

###
### Load Environments
###

@load "env.jld2"

println("Started at ", Dates.Time(Dates.now()), ".")

###
### Marginal Posterior Histograms
###

## species-level parameters
for j in 1:J
    ## set-up layout
    diagPlot = Figure(resolution = (1300, 1300))
    gl = diagPlot[1, 1] = GridLayout()

    ## alpha histogram
    alphahist = Axis(gl[1, 1], ylabel = "Frequency", xlabel = "α", title = string("Marginal Posterior Histograms for ", species[j]))
    hist!(vec(samDict["alpha"][j, :]), bins = 50, color = ("red", 0.5))
    hist!(vec(spamDict["alpha"][j, :]), bins = 50, color = ("blue", 0.5))

    ## β₁ histogram
    beta1hist = Axis(gl[2, 1], ylabel = "Frequency", xlabel = "β₁")
    hist!(vec(samDict["beta"][1, j, :]), bins = 50, color = ("red", 0.5))
    hist!(vec(spamDict["beta"][1, j, :]), bins = 50, color = ("blue", 0.5))

    ## β₂ histogram
    beta2hist = Axis(gl[3, 1], ylabel = "Frequency", xlabel = "β₂")
    hist!(vec(samDict["beta"][2, j, :]), bins = 50, color = ("red", 0.5))
    hist!(vec(spamDict["beta"][2, j, :]), bins = 50, color = ("blue", 0.5))

    ## output graphic
    save("sahtmp.pdf", diagPlot)
    append_pdf!(string("diagHist_", k, ".pdf"), "sahtmp.pdf", cleanup = true)
end

## global parameters
diagPlotg = Figure(resolution = (1300, 900))
gl = diagPlotg[1, 1] = GridLayout()
axs2 = Axis(gl[1, 1], xlabel = "σ²", ylabel = "Frequency")
hist!(samDict["s2"][:], bins = 50, color = ("red", 0.5))
hist!(spamDict["s2"][:], bins = 50, color = ("blue", 0.5))
axth0 = Axis(gl[2, 1], xlabel = "θ₀", ylabel = "Frequency")
hist!(spamDict["theta0"][:], bins = 50, color = ("blue", 0.5))
axth1 = Axis(gl[3, 1], xlabel = "θ₁", ylabel = "Frequency")
hist!(spamDict["theta1"][:], bins = 50, color = ("blue", 0.5))
save("sahtmp.pdf", diagPlotg)
append_pdf!(string("diagHist_", k, ".pdf"), "sahtmp.pdf", cleanup = true)
println("Finished marginal posterior histograms.")

###
### Trace Plots
###

## species-specific
for j in 1:J
    tmpP = Figure(resolution = (1300, 900))
    gl = tmpP[1, 1] = GridLayout()

    axbeta1 = Axis(gl[1, 1], title = string("Trace Plots for ", species[j]), xlabel = "Iteration", ylabel = "β₁")
    lines!((nburn+1):nmcmc, samDict["beta"][1, j, :], color = ("red", 0.5), label = "Non-preferential")
    lines!((nburn+1):nmcmc, spamDict["beta"][1, j, :], color = ("blue", 0.5), label = "Preferential")
    axislegend("Model")

    axbeta2 = Axis(gl[2, 1], xlabel = "Iteration", ylabel = "β₂")
    lines!((nburn+1):nmcmc, samDict["beta"][2, j, :], color = ("red", 0.5))
    lines!((nburn+1):nmcmc, spamDict["beta"][2, j, :], color = ("blue", 0.5))

    axalpha = Axis(gl[3, 1], xlabel = "Iteration", ylabel = "α")
    lines!((nburn+1):nmcmc, samDict["alpha"][j, :], color = ("red", 0.5))
    lines!((nburn+1):nmcmc, spamDict["alpha"][j, :], color = ("blue", 0.5))

    save("tmpTraces.pdf", tmpP)
    append_pdf!(string("traces_", k, ".pdf"), "tmpTraces.pdf", cleanup = true)
end

## global parameters
tmpP = Figure(resolution = (1300, 900))
gl = tmpP[1, 1] = GridLayout()
axs2 = Axis(gl[1, 1], xlabel = "Iteration", ylabel = "σ²", title = "Site-level Parameters")
lines!((nburn+1):nmcmc, samDict["s2"][:], color = ("red", 0.5), label = "Non-preferential")
lines!((nburn+1):nmcmc, spamDict["s2"][:], color = ("blue", 0.5), label = "Preferential")
axislegend("Model")

axth1 = Axis(gl[2, 1], xlabel = "Iteration", ylabel = "θ₀")
lines!((nburn+1):nmcmc, spamDict["theta0"][:], color = ("blue", 0.5))

axth2 = Axis(gl[3, 1], xlabel = "Iteration", ylabel = "θ₁")
lines!((nburn+1):nmcmc, spamDict["theta1"][:], color = ("blue", 0.5))
save("tmpTraces.pdf", tmpP)
append_pdf!(string("traces_", k, ".pdf"), "tmpTraces.pdf", cleanup = true)

tmpP = Figure(resolution = (1300, 900))
gl = tmpP[1, 1] = GridLayout()
axs = Axis(gl[1, 1], xlabel = "Iteration", ylabel = "̃λ", title = "Trace Plot of Abundance Threshold")
lines!((nburn+1):nmcmc, spamDict["lamThresh"][:], color = ("blue", 0.5), label = "Preferential")
axislegend("Model")
save("lamThresh_trace.png", tmpP)
println("Finished trace plots.")

###
### Plot Estimated Abundance
###

lamPointSam = zeros(T, J)
lamLBSam = zeros(T, J)
lamUBSam = zeros(T, J)
lamPointSpam = zeros(T, J)
lamLBSpam = zeros(T, J)
lamUBSpam = zeros(T, J)

for j in 1:J
    for t in 1:T
        lamPointSam[t, j] = median(samDict["lam"][t, j, :])
        lamLBSam[t, j] = quantile(samDict["lam"][t, j, :], 0.25)
        lamUBSam[t, j] = quantile(samDict["lam"][t, j, :], 0.75)
        lamPointSpam[t, j] = median(spamDict["lam"][t, j, :])
        lamLBSpam[t, j] = quantile(spamDict["lam"][t, j, :], 0.25)
        lamUBSpam[t, j] = quantile(spamDict["lam"][t, j, :], 0.75)
    end
end

nyears = length(years)
yearsAug = minimum(years):(maximum(years)+1)
vdays = [(yearsAug[i] - minimum(years))*365 for i in 1:length(yearsAug)]
midpoints = vdays[2:end] - 365/2*ones(nyears)

## growth rates gt
gtSam = zeros(T, J)
gtLBSam = zeros(T, J)
gtUBSam = zeros(T, J)
gtSpam = zeros(T, J)
gtLBSpam = zeros(T, J)
gtUBSpam = zeros(T, J)
for t in 1:T
    for j in 1:J 
        gtSam[t, j] = mean(samDict["gt"][t, j, :])
        gtLBSam[t, j] = quantile(samDict["gt"][t, j, :], 0.25)
        gtUBSam[t, j] = quantile(samDict["gt"][t, j, :], 0.75)
        gtSpam[t, j] = mean(spamDict["gt"][t, j, :])
        gtLBSpam[t, j] = quantile(spamDict["gt"][t, j, :], 0.25)
        gtUBSpam[t, j] = quantile(spamDict["gt"][t, j, :], 0.75)
    end
end

for j in 1:J
    tmpP = Figure(resolution = (1300, 900), fontsize = 19)
    gl = tmpP[1, 1] = GridLayout()

    axest = Axis(gl[1, 1], title = string(species[j], " at ", sites[k]), xlabel = "Days", ylabel = "Relative Abundance", xgridvisible = false, ygridvisible = false, titlefont = "TeX Gyre Heros Bold Italic Makie")
    scatter!((1:T)[findall(removedObs.==1)], Y[findall(removedObs.==1), j], strokewidth = 2, strokecolor = :black, markersize = 8, color = :white)
    scatter!((1:T)[findall(T_ind.==1)], Y[findall(T_ind.==1), j], color = :black, markersize = 6)
    
    band!(1:T, lamLBSam[:, j], lamUBSam[:, j], color = ("red", 0.25))
    band!(1:T, lamLBSpam[:, j], lamUBSpam[:, j], color = ("blue", 0.25))
    lines!(1:T, lamPointSam[:, j], color = ("red", 0.75), label = "Non-preferential", linewidth = 2)
    lines!(1:T, lamPointSpam[:, j], color = ("blue", 0.75), label = "Preferential", linewidth = 2, linestyle = :dashdotdot)
    vlines!(vdays, color = (:black, 0.5))
    axislegend("Model")

    axgt = Axis(gl[2, 1], xlabel = "Days", ylabel = "gⱼ(t)", xgridvisible = false, ygridvisible = false)
    hlines!(0, color = (:black, 0.5), linestyle = :dash)
    band!(1:T, gtLBSam[:, j], gtUBSam[:, j], color = ("red", 0.25))
    band!(1:T, gtLBSpam[:, j], gtUBSpam[:, j], color = ("blue", 0.25))
    lines!(1:T, gtSam[:, j], color = ("red", 0.75), linewidth = 2)
    lines!(1:T, gtSpam[:, j], color = ("blue", 0.75), linewidth = 2, linestyle = :dashdotdot)
    vlines!(vdays, color = (:black, 0.5))

    save("tmpTraces.pdf", tmpP)
    append_pdf!(string("est_", k, ".pdf"), "tmpTraces.pdf", cleanup = true)
end
println("Finished abundance plots.")

###
### Ridge Plot Data
###

## get posterior estimates for each iteration for each year
samFirsts = zeros(nyears, J, nmcmc-nburn)
spamFirsts = zeros(nyears, J, nmcmc-nburn)
for yr in 1:nyears
    tmpRange = ((years[yr] - minimum(years))*365 + 1):((years[yr] - minimum(years))*365 + 365)
    for j in 1:J
        for i in 1:(nmcmc - nburn)
            for t in tmpRange

                ## first day we see abundance exceed 1
                if samFirsts[yr, j, i]==0 && samDict["lam"][t, j, i] > 1
                    samFirsts[yr, j, i] = t
                end
                if spamFirsts[yr, j, i]==0 && spamDict["lam"][t, j, i] > 1
                    spamFirsts[yr, j, i] = t
                end
            end
        end
    end
end

## restructure variables for R
years = collect(years)

## new ridge plot
@rput years samFirsts spamFirsts species J nyears phenometric k pathName
R"""
source(paste0(pathName, "/ridgePlot.R")) # plots multiple species
suppressWarnings(suppressMessages(ridge_plot(years, samFirsts, spamFirsts, species, J, nyears, phenometric, k)))
"""
println("Plotted Ridges.")

println("Finished at ", Dates.Time(Dates.now()), ".")