library(ggplot2)

#Set directory

setwd("Your directory here")


#Load in CNOG file and differences reference file

compare <- read.csv("Ara-3_CNV_differences_joesph_subset.csv") #Reference file


Ara <- read.csv("Ara-3_40000gen.txt", sep = "\t") #CNOG File


#Go through the reference file and find things to analyze
compareClean <- compare %>%
  filter(length > 100) %>% #Minimum size
  filter(sample == "Ara-3_40000gen") #Choose sample


Ara <- Ara[Ara$Type == "CDS",] #Select type of event

#Clean data so it will graph

Ara$Locus <- substr(Ara$Locus, 5, nchar(Ara$Locus)) #Select number from locus

Ara$Locus <- as.numeric(Ara$Locus)

Ara$CN_HMM <- as.numeric(Ara$CN_HMM)

#This filters out areas where CNOG thinks it's normal to allow for better comparison to reference file, as it also doesn't include normal. It's entirely optional
AraClean <- Ara %>% 
  filter(CN_HMM != 1)

#Create graphs


#CNOG
ggplot(data= AraClean,aes(x=Left, y= CN_HMM, color = "orange")) + 
  geom_point() +
  #scale_color_gradient(low="orange", high="black") +
  ggtitle("Ara-3_40000 CNOGPro") +
  ylab("Copy Number") +
  xlab("Position")

#Reference
ggplot(data= compareClean,aes(x=start, y= CNV, color = "orange", size = length)) + 
  geom_point() +
 # scale_color_gradient(low="orange", high="black") +
  ggtitle("Ara-3_40000 Known") +
  ylab("Copy Number") +
  xlab("Position") 
