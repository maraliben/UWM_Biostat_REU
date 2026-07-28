install.packages("PRROC")
require(PRROC)

data_results <- matrix(0, nrow =  4 * 6 * 5 * 20, ncol = 9)
index <- 1
colnames(data_results) <- c("d", "omega", "p", "iteration", "AUPRC", "Coverage", "Power", "Size", "FDR")
for (d in c(1,4,8,12)) {
  for (omega in c(0.1,0.2,0.4,0.5,0.7,0.8)) {
    for (p in c(0.1,0.3,0.5,0.7,0.9)) {
      for (i in 1:20) {
        data_results[index, 1:4] <- c(d, omega, p, i)
        dir <- paste0("$RESULTS_DIR /sim_results_d", d, "_w", omega, "_p", p, "_", i)
        ###### use in the case that some sims didn't properly run
        # if (!file.exists(dir)) {
        #   next
        # }
        # if (!file.exists(paste0(dir, "/pip.csv"))) {
        #   next
        # }
        causal_variants <- scan(paste0(dir, "/sim1000G_cat_causalvar.txt"), sep = " ", quiet = TRUE)
        pip <- read.csv(paste0(dir, "/pip.csv"))
        if (file.size(paste0(dir, "/credible_set.txt")) == 0) { # if no credible sets are identified
          cred_sets <- matrix(0, nrow = 1, ncol = 1)
          matching <- 0
          sets_with_causal <- 0
        } else {
          cred_sets <- read.table(paste0(dir, "/credible_set.txt"), header = FALSE, sep = " ", fill = TRUE) + 1
          matching <- sum(unlist(cred_sets) %in% causal_variants) # how many causal variants are identified by BEATRICE
          condit_prob <- read.table(paste0(dir, "/conditional_credible_variants_probability.txt"), header = FALSE, sep = " ", fill = TRUE)
          sets_with_causal <- rep(F, dim(cred_sets)[1])
          for (j in 1:dim(cred_sets)[1]) { # number of variants within each credible set
            matching_in_set <- sum(cred_sets[j,] %in% causal_variants)
            binary <- ifelse(matching_in_set > 0, T, F)
            sets_with_causal[j] <- binary
          }
        }
        
        # AUPRC
        pr <- pr.curve(scores.class0=pip[causal_variants,2],
                       scores.class1=pip[-causal_variants,2],
                       curve=T)
        data_results[index, 5] <- pr$auc.integral
        
        # Coverage
        data_results[index, 6] <- sum(sets_with_causal) / dim(cred_sets)[1]
        
        # Power
        data_results[index, 7] <- matching / d
        
        # Size
        data_results[index, 8] <- sum(!is.na(cred_sets)) / dim(cred_sets)[1]
        
        # FDR
        data_results[index, 9] <- matching / sum(!is.na(cred_sets))
        
        index <- index + 1
        # print(paste(d, omega, p, i))
      }
    }
  }
}

# preliminary graphs
d_graph <- matrix(0, ncol = 4, nrow = 4)
rownames(d_graph) <- c(1, 4, 8, 12)
colnames(d_graph) <- c("AUPRC", "Coverage", "Power", "Size")
d_graph[1,] <- colMeans(data_results[which(data_results[,1] == 1,),5:8])
d_graph[2,] <- colMeans(data_results[which(data_results[,1] == 4,),5:8])
d_graph[3,] <- colMeans(data_results[which(data_results[,1] == 8,),5:8])
d_graph[4,] <- colMeans(data_results[which(data_results[,1] == 12,),5:8])
for (i in 1:(dim(d_graph)[2] - 1)) {
  plot(d_graph[,i], type = "l", xlab = "d", ylab = colnames(d_graph)[i], ylim = c(0,1))
}
plot(d_graph[,4], type = "l", xlab = "d", ylab = colnames(d_graph)[4], ylim = c(0,10))

omega_graph <- matrix(0, ncol = 4, nrow = 6)
rownames(omega_graph) <- c(0.1, 0.2, 0.4, 0.5, 0.7, 0.8)
colnames(omega_graph) <- c("AUPRC", "Coverage", "Power", "Size")
omega_graph[1,] <- colMeans(data_results[which(data_results[,2] == 0.1,),5:8])
omega_graph[2,] <- colMeans(data_results[which(data_results[,2] == 0.2,),5:8])
omega_graph[3,] <- colMeans(data_results[which(data_results[,2] == 0.4,),5:8])
omega_graph[4,] <- colMeans(data_results[which(data_results[,2] == 0.5,),5:8])
omega_graph[5,] <- colMeans(data_results[which(data_results[,2] == 0.7,),5:8])
omega_graph[6,] <- colMeans(data_results[which(data_results[,2] == 0.8,),5:8])
for (i in 1:(dim(omega_graph)[2] - 1)) {
  plot(omega_graph[,i], type = "l", xlab = "omega", ylab = colnames(omega_graph)[i], ylim = c(0,1))
}
plot(omega_graph[,4], type = "l", xlab = "omega", ylab = colnames(omega_graph)[4], ylim = c(0,10))

p_graph <- matrix(0, ncol = 4, nrow = 5)
rownames(p_graph) <- c(0.1, 0.3, 0.5, 0.7, 0.9)
colnames(p_graph) <- c("AUPRC", "Coverage", "Power", "Size")
p_graph[1,] <- colMeans(data_results[which(data_results[,3] == 0.1,),5:8])
p_graph[2,] <- colMeans(data_results[which(data_results[,3] == 0.3,),5:8])
p_graph[3,] <- colMeans(data_results[which(data_results[,3] == 0.5,),5:8])
p_graph[4,] <- colMeans(data_results[which(data_results[,3] == 0.7,),5:8])
p_graph[5,] <- colMeans(data_results[which(data_results[,3] == 0.9,),5:8])
for (i in 1:(dim(p_graph)[2] - 1)) {
  plot(p_graph[,i], type = "l", xlab = "p", ylab = colnames(p_graph)[i], ylim = c(0,1))
}
plot(p_graph[,4], type = "l", xlab = "p", ylab = colnames(p_graph)[4], ylim = c(0,10))

write.table(data_results, file = "$RESULTS_DIR /data_results.txt", row.names = F, sep = " ")
