SIM_DIR="/Users/simonkopischke/Desktop/REU Biostats Project"
BEATRICE_DIR="/Users/simonkopischke/Desktop/REU Biostats Project/Beatrice-Finemapping"
RESULTS_DIR="/Users/simonkopischke/Desktop/REU Biostats Project/new_model"
Z_DIR=${SIM_DIR}
LD_DIR=${SIM_DIR}

# d in 1 4 8 12
# omega in 0.1 0.2 0.4 0.5 0.7 0.8
# p in 0.1 0.3 0.5 0.7 0.9
# i in {11..20}

for d in 1 4 8 12
do
	for omega in 0.1 0.2 0.4 0.5 0.7 0.8
	do
		for p in 0.1 0.3 0.5 0.7 0.9
		do
			for i in {11..20}
			do
				if [ -e "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}/credible_set.txt" ]
				then
					continue
				fi

				echo d${d}_w${omega}_p${p}_${i}
				
				cd "$SIM_DIR"
				Rscript "New Performance Analysis_test.R" ${d} ${omega} ${p}

				cd "$RESULTS_DIR"
				mkdir -p "sim_results_d${d}_w${omega}_p${p}_${i}"

				cd "$SIM_DIR"

 				ls -lh "${SIM_DIR}/sim1000G_cat_zscores.csv"
 				ls -lh "${SIM_DIR}/sim1000G_cat_LD.csv"
				ls -lh "${SIM_DIR}/sim1000G_cat_causalvar.txt"
		
				echo "renaming causal_v files and moving..."
				mv "${SIM_DIR}/sim1000G_cat_causalvar.txt" "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}/sim1000G_cat_causalvar.txt"

				cd "$BEATRICE_DIR"
		
				ls -lh "${Z_DIR}/sim1000G_cat_zscores.csv"
				ls -lh "${LD_DIR}/sim1000G_cat_LD.csv"

				python3 beatrice.py --z "${Z_DIR}/sim1000G_cat_zscores.csv" --LD "${LD_DIR}/sim1000G_cat_LD.csv" --N 5000 --target "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}"

				# removing files
				rm -f "${SIM_DIR}/sim1000G_cat_zscores.csv"
 				rm -f "${SIM_DIR}/sim1000G_cat_LD.csv"
			done
		done
	done
done
