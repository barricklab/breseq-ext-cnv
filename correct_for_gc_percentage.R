library(tidyverse)
library(readr)
library(ggplot2)

plot_theme = theme_bw()

X = read_tsv("NC_001416-0.coverage.tab")
seq_length = nrow(X)

seq = read_tsv("lambda.1-2.fna")
colnames(seq) = c("seq")
seq = paste0(seq$seq, collapse="")


gc = strsplit(seq, "")[[1]]
gc_bin = rep(0, length(gc))
gc_bin[gc %in% c("C", "G")] = 1 
  
# Double to prevent overflows
gc_bin = c(gc_bin,gc_bin,gc_bin)

# Mask regions with redundant coverage

X$unique_cov = X$unique_top_cov + X$unique_bot_cov
X$redundant_cov = X$redundant_top_cov + X$redundant_bot_cov

X = X %>% filter(redundant_cov==0)

ggplot(X, aes(x=position, y=unique_cov)) +
  geom_point() + theme_bw()


fragment_length = 200

### calculate gc given a fragment length
gc_percentages = data.frame(position=1:seq_length)
gc_percentages$gc_percent = 0

for(i in 1:seq_length) {
  gc_percentages$gc_percent[i] = sum(gc_bin[(seq_length+i-(fragment_length/2)+1):(seq_length+i+fragment_length/2)]) / fragment_length
}

ggplot(gc_percentages, aes(x=position, y=gc_percent)) +
  geom_point() + theme_bw()

X = X %>% left_join(gc_percentages, by="position")

ggplot(X, aes(x=unique_cov, y=gc_percent)) +
  geom_point(alpha=0.05) +
  plot_theme

gc_percent_df = X %>% group_by(gc_percent) %>% summarize(mean_cov = median(unique_cov))
write_csv(gc_percent_df, "GCpercent.csv")

ggplot(gc_percent_df, aes(x=gc_percent, y=mean_cov)) +
  geom_point() +
  plot_theme

