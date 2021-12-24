#!/bin/bash

declare -a selmethod=("herding")
declare -a nprots=(1)
declare -a subjects=(1)

#declare -a subjects=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)

# CAPSULE NETWORK
#	self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=0)
#	self.primarycaps = PrimaryCapsule(in_channels=64, out_channels=8*16, dim_caps=8, kernel_size=9, stride=2, padding=0)
#	self.digitcaps = DenseCapsule(in_num_caps=832, in_dim_caps=8, out_num_caps=num_output_capsules, out_dim_caps=16, routings=routings)


for alg in "${selmethod[@]}"
do
	for p in "${nprots[@]}"
	do
		for subj in "${subjects[@]}"
		do
	
			for iter in {1..5}; do
				seed=$(($((100*$iter))+123));
			
				echo "Subject:" $subj "Iteration:" $iter " alg:" $alg " percentage:" $p " seed:" $seed
				
				python3 class_incremental_cosine.py --dataset DEAP --nb_cl_fg 2 --nb_cl 1 --nn capsnet --subject $subj --fix_budget True --nb_protos $p --is_alg $alg --preprocess_format rnn --dimention valence --resume --seed $seed --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt

				echo ""
				echo ""
			done
		done
	done
done

