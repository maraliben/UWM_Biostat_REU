SIM_DIR="path of R file and causalvar.txt"
BEATRICE_DIR="path of /Beatrice-Finemapping"
RESULTS_DIR="desired path of results folders"
Z_DIR=${SIM_DIR}
LD_DIR=${SIM_DIR}

for i in {1..20}
do
	for d in 1 4 8 12
	do
		for omega in 0.1 0.2 0.4 0.5 0.7 0.8
		do
			for p in 0.1 0.3 0.5 0.7 0.9
			do
				if [ -e "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}/credible_set.txt" ]
				then
					continue
				fi

				echo d${d}_w${omega}_p${p}_${i}
				
				cd "$SIM_DIR"
				Rscript "Simulate_ZScores_LD_categorical.R" ${d} ${omega} ${p}

				cd "$RESULTS_DIR"
				mkdir -p "sim_results_d${d}_w${omega}_p${p}_${i}"

				cd "$SIM_DIR"

 				ls -lh "${Z_DIR}/sim1000G_cat_zscores.csv"
 				ls -lh "${LD_DIR}/sim1000G_cat_LD.csv"
				ls -lh "${SIM_DIR}/sim1000G_cat_causalvar.txt"
		
				echo "moving causal_v file..."
				mv "${SIM_DIR}/sim1000G_cat_causalvar.txt" "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}/sim1000G_cat_causalvar.txt"

				cd "$BEATRICE_DIR"
		
				ls -lh "${Z_DIR}/sim1000G_cat_zscores.csv"
				ls -lh "${LD_DIR}/sim1000G_cat_LD.csv"

				python3 beatrice.py --z "${Z_DIR}/sim1000G_cat_zscores.csv" --LD "${LD_DIR}/sim1000G_cat_LD.csv" --N 5000 --target "${RESULTS_DIR}/sim_results_d${d}_w${omega}_p${p}_${i}"

				# removing files
				rm -f "${Z_DIR}/sim1000G_cat_zscores.csv"
 				rm -f "${LD_DIR}/sim1000G_cat_LD.csv"
			done
		done
	done
done
