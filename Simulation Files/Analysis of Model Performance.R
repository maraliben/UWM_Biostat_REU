#setwd("")
library(sim1000G)
library(gplots)
library(VariantAnnotation)
library(dplyr)
library(vcfR)

#load(""path for genotype_10000_R file"")
genotype_10000 <- genotype
new_geno <- sample_n(as.data.frame(genotype_10000), 5000)
new_geno <- new_geno[, sample(ncol(new_geno), size = 1000)]
n <- 5000
m <- ncol(new_geno)
args <- commandArgs(trailingOnly = TRUE)
d <- as.numeric(args[1])
omega_2 <- as.numeric(args[2])
p <- as.numeric(args[3])
causal_variants <- sample(m,d)

beta <- matrix(rnorm(d, mean = 0, sd = 1), ncol = 1, nrow = d)

X <- new_geno
X_C <- as.matrix(X[,causal_variants])
X_NC <- as.matrix(X[,-causal_variants])

g_C <- X_C %*% beta
g_NC <- matrix(rnorm(n, mean = 0, sd = sqrt((X_NC %*% t(X_NC)) / (m - d))), nrow = n, ncol = 1)
epsilon <- rnorm(n, mean = 0, sd = sqrt(1 - omega_2))

normalized_y <- as.numeric(sqrt((p*omega_2)/var(g_C)))*g_C + as.numeric(sqrt(((1-p)*omega_2)/(var(g_NC))))*g_NC + epsilon
normalized_y <- exp(normalized_y) / (1 + exp(normalized_y))
for (i in 1:n) {
  normalized_y[i] <- sample(x = c(0,1), size = 1, prob = c(1 - normalized_y[i], normalized_y[i]))
}

beta_hat <- rep(0, m)
z <- beta_hat
for (i in 1:m) {
  model <- lm(normalized_y ~ X[,i])
  beta_hat[i] <- summary(model)$coef[2,1]
  z[i] <- beta_hat[i] / summary(model)$coef[2,2]
}

X <- as.matrix(X)
LD <- cor(X)

write.table(z, file = "sim1000G_cat_zscores.csv", col.names = F, sep = " ")
write.table(LD, file = "sim1000G_cat_LD.csv", row.names = F, col.names = F, sep = " ")
write.table(causal_variants, file = "sim1000G_cat_causalvar.txt", row.names = F, col.names = F, sep = " ")
