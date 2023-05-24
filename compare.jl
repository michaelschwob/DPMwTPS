###
### Script for Fitting the Preferential and Non-preferential models to NEON data
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

println("Time started: ", Dates.Time(Dates.now()), ".")

## set seed
Random.seed!(702)

## Get MCMC Functions
include("sam.mcmc.jl")
include("spam.mcmc.probit.jl")

nmcmc = 500000
nburn = Int(ceil(0.8*nmcmc))

## Read in data
mosquito_data = CSV.read("Mosquito_Count_iter.csv", DataFrame, missingstring = "NA") 
mosquito_data = mosquito_data |> @mutate(Prop = _.SubsetWeight/_.TotalWeight) |> DataFrame
mosquito_data[isnan.(mosquito_data.Prop), :15] .= 0

weather_data = CSV.read("Weather_iter.csv", DataFrame, missingstring = "NA")
sites = unique(mosquito_data.Site)

## sites we want to analyze
sites = ["UNDE"]

## set Directory
#mkdir("Outputw0") # for scenario 1
mkdir("Outputwo0") # for scenario 2

###
### Average >24hr TrapCounts Observations
###

averagedObs = mosquito_data[1, :] |> DataFrame # initialize array for averaged observations
for i in 1:size(mosquito_data)[1] # for each observation

    if mosquito_data.TrapHours[i] > 24 # if traps were left out for more than 24hrs

        numHours = mosquito_data.TrapHours[i] # number of hours in trap
        numDays = ceil(numHours/24) # number of days we need to average

        for j in 1:(numDays-1) # for each day we need to average over fully (doesn't include last day)
            push!(averagedObs, mosquito_data[i, :]) # add an averaged observation that needs modification
            averagedObs[size(averagedObs)[1], :].Count = round(mosquito_data.Count[i]/numHours*24) # change count to average count per day
            averagedObs[size(averagedObs)[1], :].TrapHours = 24 # left out for entire day
            averagedObs[size(averagedObs)[1], :].DOY = mosquito_data.DOY[i] + (j-1) # shift DOY
        end

        ## for last day
        j = numDays
        push!(averagedObs, mosquito_data[i, :])
        remHrs = numHours - 24*(j-1)
        averagedObs[size(averagedObs)[1], :].Count = round(mosquito_data.Count[i]/numHours*remHrs) # change count to average count per day
        averagedObs[size(averagedObs)[1], :].TrapHours = remHrs # left out for remaining hours
        averagedObs[size(averagedObs)[1], :].DOY = mosquito_data.DOY[i] + j - 1 # shift DOY
    end
end
averagedObs = averagedObs[2:end, :] # remove initial row

mosquito_data = mosquito_data |> @filter(_.TrapHours <= 24) |> DataFrame # remove counts greater than 24 hours
mosquito_data = vcat(mosquito_data, averagedObs)

###
### Covariate Convolution
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

println("Finished setup.")

###
### Structure Data
###

k = 1
weather_tmp = weather_data |> @filter(_.Site == sites[k]) |> DataFrame
mosquito_tmp = mosquito_data |> @filter(_.Site == sites[k]) |> DataFrame

if sites[k] == "UNDE"
    species = ["Aedes canadensis", "Aedes excrucians", "Aedes punctor"]
end
J = length(species)

if sites[k] == "UNDE"
    years = 2016:2019
end
nyears = length(years)
T = 365*nyears

## design matrix
GDD = weather_tmp |> @filter(_.Year in years) |> @select(:GDD) |> @dropna() |> DataFrame |> Matrix
PPT = weather_tmp |> @filter(_.Year in years) |> @select(:PPT) |> @dropna() |> DataFrame |> Matrix
theta_basis = 30
GDD_MA = moving_average(theta_basis, GDD)
GDD_MA = GDD_MA[(length(GDD_MA)-T+1):end]
X = [ones(T) GDD_MA]
CSV.write(string("X.csv"), Tables.table(X), header = false)

## species_data
species_data = mosquito_tmp |> @filter(_.SciName in species) |> @select(:SciName, :Year, :DOY, :Count, :TrapHours, :Prop) |> DataFrame
species_data = groupby(species_data, [:SciName, :Year, :DOY])
species_data = combine(species_data, :Count => sum => :Count, :TrapHours => sum => :TrapHours, :Prop => mean => :Prop)
species_data.DOS = ones(size(species_data)[1])

for i in 1:size(species_data)[1]
    year = species_data.Year[i]
    doy = species_data.DOY[i]
    if year == 2014
        species_data.DOS[i] = doy + 0*365
    end
    if year == 2015
        species_data.DOS[i] = doy + 1*365
    end
    if year == 2016
        species_data.DOS[i] = doy + 2*365
    end
    if year == 2017
        species_data.DOS[i] = doy + 3*365
    end
    if year == 2018
        species_data.DOS[i] = doy + 4*365
    end
    if year == 2019
        species_data.DOS[i] = doy + 5*365
    end
end

species_data = sort(species_data, [:SciName, :DOS])

## Y, H, and W
Y = zeros(T, J)
H = zeros(T, J)
W = zeros(T, J)
T_ind = zeros(T)
r_lb = (minimum(years) - 2014)*365 + 1
r_ub = (maximum(years) - 2014)*365 + 365

for t in r_lb:r_ub

    ## for Y
    tmpDF = species_data |> @filter(_.DOS == t) |> DataFrame
    for j in 1:J
        if species[j] in tmpDF.SciName
            Y[t - r_lb + 1, j] = tmpDF.Count[findfirst(x -> x == species[j], tmpDF.SciName)]
        end
    end

    ## for T_ind, H, and W
    if t in species_data.DOS
        T_ind[t - r_lb + 1] = 1
        for j in 1:J
            if species[j] in tmpDF.SciName
                H[t-r_lb+1, j] = tmpDF.TrapHours[findfirst(x -> x == species[j], tmpDF.SciName)]/24
                W[t-r_lb+1, j] = tmpDF.Prop[findfirst(x -> x == species[j], tmpDF.SciName)]
            end
        end
    end
end

###
### Remove 0 counts across all species (optional for scenario 2)
###

removedObs = zeros(T) # vector to plot actual observations again
for t in 1:T
    if T_ind[t] == 1
        removedObs[t] = 1
    end
    if sum(Y[t, :]) == 0 # if no species had observations...
        T_ind[t] = 0 # ... remove 0 counts
    end
end

## save data
CSV.write(string("Tind.csv"), Tables.table(T_ind), header = false)

###
### Compute Phenometric of First Day of Population Growth
###

daysObs = findall(T_ind.==1) # list of days observed
YObs = Y[daysObs, :] # counts observed

phenometric = zeros(J)
for j in 1:J # for each species
    tmpVec = zeros(nyears) # initialize first day in years
    for yr in 1:nyears # for each year

        ## get this year's range
        dayRange = ((years[yr] - minimum(years))*365 + 1):((years[yr] - minimum(years))*365 + 365) # obtain range of days

        ## find which indices of daysObs & YObs are in this year's range
        legalIdx = vec([0]) # initialize vector of indices
        for t in 1:length(daysObs)
            if daysObs[t] in dayRange
                append!(legalIdx, t)
            end
        end
        legalIdx = legalIdx[2:end] # remove initial 0

        ## if first year
        if yr == 1
            for t in legalIdx # for each observed day in range 
                if YObs[t+1, j] > YObs[t, j] && tmpVec[yr] == 0 # if next day is more than today
                    tmpVec[yr] = daysObs[t+1] # next day is first day of population growth
                end
            end
        else
            for t in legalIdx # for each observed day in range 
                if YObs[t, j] > YObs[t-1, j] && tmpVec[yr] == 0 # if next day is more than today
                    tmpVec[yr] = daysObs[t] - 365*(yr-1) # next day is first day of population growth
                end
            end
        end
    end
    phenometric[j] = mean(tmpVec) # average phenometric across years
end

###
### Fit Model
###

## Run Preferential Model (SPAM)
println("SPAM:")
spam_out = spamMCMC(Y, T_ind, X, H, W, nmcmc, nburn)

## Run Non-preferential model (SAM)
println("SAM:")
sam_out = samMCMC(Y, T_ind, X, H, W, nmcmc, nburn)

###
### Write Dictionaries & Save Environment
###

samDict = Dict("alpha" => sam_out[1], "beta" => sam_out[2], "lam" => sam_out[3], "s2" => sam_out[4], "gt" => sam_out[5])
sam_out = nothing

spamDict = Dict("alpha" => spam_out[1], "beta" => spam_out[2], "lam" => spam_out[3], "s2" => spam_out[4], "gt" => spam_out[5], "theta0" => spam_out[6], "theta1" => spam_out[7], "lamThresh" => spam_out[8])
spam_out  =  nothing

#@save "Outputw0/env.jld2"
@save "Outputwo0/env.jld2"

###
### End File
###

println("Time ended: ", Dates.Time(Dates.now()), ".")