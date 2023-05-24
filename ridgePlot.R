###
### Plot Ridge Plots Using Inference From Server Runs
###

library(ggplot2)
library(ggridges)
library(tidyverse)
library(ggpubr)
library(forcats)

sites <- c("UNDE", "NIWO")

ridge_plot <- function(years, samFirsts, spamFirsts, species, J, nyears, phenometric, k){

  maxR = max(phenometric) + 25 # limit of ridge plot

  ###
  ### Ridge Plots for Competing Models on the same line
  ###

  for(t in 1:nyears){ # for each year

    ## get range of days
    tmp.range <- ((years[t] - min(years))*365 + 1):((years[t] - min(years))*365 + 365)

    ## ensure that vectors are proper lengths
    samvec = c(samFirsts[t, 1, ], samFirsts[t, 2, ], samFirsts[t, 3, ]) # only keep phenometrics that are the in proper year
    spamvec = c(spamFirsts[t, 1, ], spamFirsts[t, 2, ], spamFirsts[t, 3, ]) # only keep phenometrics that are in the proper year

    specvec = c(rep(species[1], length(samFirsts[t, 1, ])), rep(species[2], length(samFirsts[t, 2, ])), rep(species[3], length(samFirsts[t, 3, ])), rep(species[1], length(spamFirsts[t, 1, ])), rep(species[2], length(spamFirsts[t, 2, ])), rep(species[3], length(spamFirsts[t, 3, ])))

    ridge.df <- data.frame(Days = c(samvec - 365*(years[t]-min(years)), spamvec - 365*(years[t]-min(years))), Species = specvec, Model = c(rep("SAM", length(samvec)), rep("SPAM", length(spamvec))))

    ridge.lines <- data.frame(Species = c(1, 2, 3), x0 = phenometric)

    ## obtain ridge plot
    tog <- ridge.df %>%
    mutate(SpeciesFct = fct_rev(as.factor(Species))) %>%
    ggplot(aes(y = SpeciesFct)) +
    geom_density_ridges(
      aes(x = Days, fill = paste(SpeciesFct, Model)), 
      alpha = .5, color = "white", from = 0, to = maxR, scale = 1, bandwidth = 3
    ) +
    labs(
      x = ifelse(t==4, "Day of the Year", ""),
      y = paste0(years[t]),
      title = ifelse(t==2, expression(paste("First Day of Population Growth at UNDE (", psi, ")")), "")
    ) +
    scale_y_discrete(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_cyclical(
      breaks = c("Aedes canadensis SAM", "Aedes canadensis SPAM"),
      labels = c(`Aedes canadensis SAM` = "Non-preferential", `Aedes canadensis SPAM` = "Preferential"), # "SAM", "SPAM"
      values = c("red", "blue", "red", "blue"),
      name = "Model", guide = "legend"
    ) +
    coord_cartesian(clip = "off") +
    theme_ridges(grid = FALSE) + 
    geom_segment(data = ridge.lines, aes(x = x0, xend = x0, y = as.numeric(Species), yend = as.numeric(Species) + 0.9), col = "black") + 
    theme_minimal() + theme(plot.title = element_text(hjust = 0.5)) + theme(axis.text.y = element_text(face = "italic"))

    assign(paste0("rdg", t), tog)
  }
  arranged <- ggarrange(rdg2, rdg3, rdg4, nrow = 3)
  arranged
  ggsave(paste0("ridgePlot_", k, ".png"), arranged, width = 10, height = 10, bg = "white")  
}