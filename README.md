# DPMwTPS
Schwob et al. (In Press) "Dynamic Population Models with Temporal Preferential Sampling to Infer Phenology," _Journal of Agricultural, Biological, and Environmental Statistics_.

## Abstract
To study population dynamics, ecologists and wildlife biologists typically use relative abundance data, which may be subject to temporal preferential sampling. Temporal preferential sampling occurs when the times at which observations are made and the latent process of interest are conditionally dependent. To account for preferential sampling, we specify a Bayesian hierarchical abundance model that considers the dependence between observation times and the ecological process of interest. The proposed model improves relative abundance estimates during periods of infrequent observation and accounts for temporal preferential sampling in discrete time. Additionally, our model facilitates posterior inference for population growth rates and mechanistic phenometrics. We apply our model to analyze both simulated data and mosquito count data collected by the National Ecological Observatory Network. In the second case study, we characterize the population growth rate and relative abundance of several mosquito species in the _Aedes_ genus.

## Files

`compare.jl`
A Julia script that prepares the NEON data and runs the preferential (SAM) and non-preferential (SPAM) MCMC algorithms.

`plotting.jl`
A Julia script that obtains the plots used for publication, as well as diagnostic plots. Refers to `ridgePlot.R`.

`ridgePlot.R`
An R script that obtains the ridge plots for the phenometrics.

`sam.mcmc.jl`
A Julia script that contains the non-preferential MCMC algorithm.

`spam.mcmc.probit.jl`
A Julia script that contains the preferential MCMC algorithm.

`>Outputw0`
Contains the output of `plotting.jl` under the first scenario in the NEON case study.

`>Outputwo0`
Contains the output of `plotting.jl` under the second scenario in the NEON case study.

`>Simulation/script.jl`
A Julia script that simulates data for the first case study and runs the SAM and SPAM MCMC algorithms contained in `sam.mcmc.jl` and `spam.mcmc.probit.jl`. This script also plots the output contained in `>Simulation` and the `>Simulation/RSME.csv` file.

## Data 

`Mosquito_Count_iter.csv`
A subset of the mosquito count data collected by NEON.

`Weather_iter.csv`
Weather data obtained from Oregon PRISM for all sites listed in `Mosquito_Count_iter.csv` throughout the study period. Used in the simulated and NEON case studies.
