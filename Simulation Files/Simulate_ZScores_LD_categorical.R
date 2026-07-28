#setwd("$SIM_DIR")
library(sim1000G)
library(gplots)
library(VariantAnnotation)
library(vcfR)
library(mvtnorm)

#load("path for genotype_10000_R file")
genotype_10000 <- genotype
new_geno <- genotype_10000[sample(nrow(genotype_10000), size = 5000),]
new_geno <- new_geno[, sample(ncol(new_geno), size = 1000)]
n <- 5000
m <- ncol(new_geno)
args <- commandArgs(trailingOnly = TRUE)
d=as.numeric(args[1])
omega_2=as.numeric(args[2])
p=as.numeric(args[3])
causal_variants <- sample(m,d)

beta <- matrix(rnorm(d, mean = 0, sd = 1), ncol = 1, nrow = d)

X <- new_geno
X <- scale(X, center = TRUE, scale = TRUE)
X_C <- as.matrix(X[,causal_variants])
X_NC <- as.matrix(X[,-causal_variants])

g_C <- X_C %*% beta
sigma_val <- (X_NC %*% t(X_NC)) / (m - d)
g_NC <- as.matrix(rmvnorm(n=1, mean=rep(0,n), sigma=sigma_val))
epsilon <- rnorm(n, mean = 0, sd = sqrt(1 - omega_2))

normalized_y <- as.numeric(sqrt((p*omega_2)/var(g_C)))*g_C + as.numeric(sqrt(((1-p)*omega_2)/(var(g_NC))))*g_NC + epsilon
normalized_y <- exp(normalized_y) / (1 + exp(normalized_y))
for (i in 1:n) { # y's must be 0 and 1
  normalized_y[i] <- sample(x = c(0,1), size = 1, prob = c(1 - normalized_y[i], normalized_y[i]))
}

X_mat <- as.matrix(X)
LD <- cor(X_mat)
z <- t(X_mat) %*% log((normalized_y+1)/(2-normalized_y))

write.table(z, file = "sim1000G_cat_zscores.csv", col.names = F, sep = " ")
write.table(LD, file = "sim1000G_cat_LD.csv", row.names = F, col.names = F, sep = " ")
write.table(causal_variants, file = "sim1000G_cat_causalvar.txt", row.names = F, col.names = F, sep = " ")
