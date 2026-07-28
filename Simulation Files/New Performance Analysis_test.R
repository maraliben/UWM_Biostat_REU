setwd("/Users/simonkopischke/Desktop/REU Biostats Project")
library(sim1000G)
library(gplots)
library(VariantAnnotation)
library(vcfR)
library(mvtnorm)
#  
load("/Users/simonkopischke/Desktop/REU Biostats Project/csv's and R objects/genotype_10000_R")
genotype_10000 <- genotype
#new_geno <- sample_n(as.data.frame(genotype_10000), 5000)
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
X_C <- as.matrix(X[,causal_variants]) # scaled causal variants
#X_C <- as.matrix(X[,causal_variants]) # scaled causal variants

X_NC <- as.matrix(X[,-causal_variants]) # scaled non-causal variants
#X_NC <- as.matrix(X[,-causal_variants]) # scaled non-causal variants

g_C <- X_C %*% beta
#dave said to normalize X's - mean 0, std dev sum 1
sigma_val <- (X_NC %*% t(X_NC)) / (m - d)

g_NC <- rmvnorm(n=1, mean=rep(0,n), sigma=sigma_val)
g_NC <- as.vector(g_NC)
g_NC <- as.matrix(g_NC)
#g_NC <- matrix(rnorm(n, mean = 0, sd = sqrt((X_NC %*% t(X_NC)) / (m - d))), nrow = n, ncol = 1)

epsilon <- rnorm(n, mean = 0, sd = sqrt(1 - omega_2))

#y <- g_C + g_NC + epsilon

normalized_y <- as.numeric(sqrt((p*omega_2)/var(g_C)))*g_C + as.numeric(sqrt(((1-p)*omega_2)/(var(g_NC))))*g_NC + epsilon


#y <- g_C + g_NC + rnorm(n, mean = 0, sd = sqrt(1 - omega_2))
normalized_y <- exp(normalized_y) / (1 + exp(normalized_y))
for (i in 1:n) {
normalized_y[i] <- sample(x = c(0,1), size = 1, prob = c(1 - normalized_y[i], normalized_y[i]))
}

X_mat <- as.matrix(X)
# y's must be 0 and 1
#z <- ginv(t(X_mat)%*%X_mat)%*%t(X_mat)%*%log((normalized_y+1)/(2-normalized_y))
z <- t(X_mat) %*% log((normalized_y+1)/(2-normalized_y))
#beta_hat <- rep(0, m)
#z_old <- beta_hat
#for (i in 1:m) {
 # model <- lm(normalized_y ~ X[,i])
#  beta_hat[i] <- summary(model)$coef[2,1]
 # z_old[i] <- beta_hat[i] / summary(model)$coef[2,2]
#}

LD <- cor(X_mat)

write.table(z, file = "sim1000G_cat_zscores.csv", col.names = F, sep = " ")
write.table(LD, file = "sim1000G_cat_LD.csv", row.names = F, col.names = F, sep = " ")
write.table(causal_variants, file = "sim1000G_cat_causalvar.txt", row.names = F, col.names = F, sep = " ")