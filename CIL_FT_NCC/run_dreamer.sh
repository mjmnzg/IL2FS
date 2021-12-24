#!/bin/bash

declare -a selmethod=("herding")
declare -a nprots=(10)
declare -a subjects=(1)

# DREAMER
#declare -a subjects=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)


# CAPSULE NETWORK
#	self.conv1 = nn.Conv2d(1, 64, kernel_size=6, stride=1, padding=0)
#	self.primarycaps = PrimaryCapsule(in_channels=64, out_channels=8*16, dim_caps=8, kernel_size=6, stride=2, padding=0)
#	self.digitcaps = DenseCapsule(in_num_caps=1888, in_dim_caps=8, out_num_caps=num_output_capsules, out_dim_caps=16, routings=routings)


# To execute LUCIR/Geodesic/Mnemonics
#	1. Select command in shell-
#	2. Remove BN layers.
#	3. Remove sampling for proposal 
#	4. Put loss1+loss2+loss3
#	4. Remove EMA 
#	5. ADD Modified Linear
#	6. Set 'args.apply_mnemonics' to True or False 
#	7. REMOVE NEM classifier
#	8. Set args.use_walign = False

# To execute PROPOSAL I
#	1. Select command in shell.
#	2. ADD BN layers.
#	3. ADD sampling for proposal
#	4. PUT loss1+loss2+loss4
#	5. ADD EMA 
#	6. Set 'apply_mnemonics' to False
#	7. ADD Modified Linear
#	8. REMOVE NEM classifier
#	9. Set args.use_walign = False

# To execute PROPOSAL II
#	1. Select command in shell.
#	2. REMOVE BN layers.
#	3. REMOVE sampling for proposal
#	4. Enable triplet loss
#	5. Enable distillation loss v2
#	6. Enable loss1 + loss2 + loss4 + loss5
#	7. ADD EMA 
#	8. Set 'args.apply_mnemonics' to False
#	9. REMOVE Modified Linear
#	10. ADD NEM classifier
#	11. Set args.use_walign = True


# To execute iCaRL/LwF.MC
#	1. Select command in shell.
#	2. REMOVE BN layers.
#	3. REMOVE sampling for proposal
#	4. ADD distillation loss v2
#	5. PUT loss1 + loss2
#	6. REMOVE EMA
#	7. REMOVE Modified Linear
#	8. Set 'args.apply_mnemonics' to False
#	9. Add NEM classifier
#	10. Set args.use_walign = False


# To execute FT
#	1. Select command in shell.
#	2. REMOVE BN layers.
#	3. REMOVE sampling for proposal
#	4. Enable loss2
#	5. REMOVE EMA 
#	6. REMOVE Modified Linear
#	7. Set 'args.apply_mnemonics' to False
#	8. Add NEM classifier
#	9. Set args.use_walign = False


# To execute Weight Aligning
#	1. Select command in shell.
#	2. REMOVE BN layers.
#	3. REMOVE sampling for proposal
#	4. Enable distillation loss v2
#	5. PUT loss1 + loss2
#	6. REMOVE EMA 
#	7. REMOVE Modified Linear
#	8. Set 'args.apply_mnemonics' to False
#	9. Add NEM classifier
#	10. Set args.use_walign = True


for alg in "${selmethod[@]}"
do
	for p in "${nprots[@]}"
	do
		for subj in "${subjects[@]}"
		do
	
			for iter in {1..1}; do
				seed=$(($((100*$iter))+123));
			
				echo "Subject:" $subj "Iteration:" $iter " alg:" $alg " prots:" $p " seed:" $seed
				
				# DREAMER-9 [FT/ICaRL/LwF/WA/PROPOSAL II] [CapsNet]
				# :
				python3 class_incremental_cosine.py --dataset DREAMER --nb_cl_fg 2 --nb_cl 1 --nn capsnet --subject $subj --fix_budget True --nb_protos $p --is_alg $alg --preprocess_format rnn --dimention discrete_emotions --resume --seed $seed --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt
				
				
				# DREAMER-9 [LUCIR/MNEMONICS] [CapsNet]
				# : 
				#python3 class_incremental_cosine.py --dataset DREAMER --nb_cl_fg 3 --nb_cl 3 --fix_budget True --nn capsnet --subject $subj --nb_protos $p --is_alg $alg --preprocess_format rnn --imprint_weights --dimention discrete_emotions --resume --rs_ratio 0.0 --less_forget --lamda 1 --adapt_lamda --seed $seed --mr_loss --dist 5 --K 1 --lw_mr 1 --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt
				
				# DREAMER-9 [PROPOSAL-I] [CapsNet]
				# : 
				#python3 class_incremental_cosine.py --dataset DREAMER --nb_cl_fg 2 --nb_cl 1 --fix_budget True --nn capsnet --subject $subj --nb_protos $p --is_alg $alg --preprocess_format rnn --dimention discrete_emotions --resume --rs_ratio 0.0 --less_forget --lamda 1 --adapt_lamda --seed $seed --mr_loss --dist 5 --K 1 --lw_mr 1 --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt
				
				# DREAMER-9 [ICaRL/LwF] [CRNN]
				#python3 class_incremental_cosine.py --dataset DREAMER --nb_cl_fg 2 --nb_cl 1 --nn crnn --subject $subj --nb_protos $p --is_alg $alg --preprocess_format cnn --dimention discrete_emotions --resume --rs_ratio 0.0 --less_forget --lamda 1 --adapt_lamda --seed $seed --mr_loss --dist 5 --K 1 --lw_mr 1 --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt
				
				# DREAMER-9 [LUCIR] [CRNN]
				#python3 class_incremental_cosine.py --dataset DREAMER --nb_cl_fg 2 --nb_cl 1 --nn crnn --subject $subj --nb_protos $p --is_alg $alg --preprocess_format cnn --dimention discrete_emotions --imprint_weights --resume --rs_ratio 0.0 --less_forget --lamda 1 --adapt_lamda --seed $seed --mr_loss --dist 5 --K 1 --lw_mr 1 --ckp_prefix seed_1993_rs_ratio_0.0_class_incremental_LFAD_cosine_cifar100 2>&1 | tee log.txt
				

				echo ""
				echo ""
			done
		done
	done
done

