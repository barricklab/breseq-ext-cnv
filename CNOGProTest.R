install.packages("CNOGpro")

#Set directory with files
setwd("Your Path Here")

library(CNOGpro)

#Load in data

#File name, gene bank reference, window length, experiment name
AraFive <- CNOGpro("Ara-3_40000gen.hits","REL606.6.gbk", 100, name = "Ara-3_4000gen")

#Normalize data
AraFive_normalized <- normalizeGC(AraFive)

#Check normalization
plotCNOGpro(AraFive_normalized)

#Bootstrap analysis
AraFive_noramlized_Bootstrap <- runBootstrap(AraFive_normalized, replicates = 1000, quantiles = c(.025,.975))

printCNOGpro(AraFive_noramlized_Bootstrap)

summaryCNOGpro(AraFive_noramlized_Bootstrap)


#Hidden Markov Model

AraFive_normalized_HMM <- runHMM(AraFive_normalized)

printCNOGpro(AraFive_normalized_HMM)

plotCNOGpro(AraFive_normalized_HMM)


#Store experiment in file

setwd("Your output directory Here")


#Generates a tab seprated file as name.txt
store(AraFive_normalized_HMM, path = "./")
