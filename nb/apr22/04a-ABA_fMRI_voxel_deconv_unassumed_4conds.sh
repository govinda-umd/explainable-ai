#!/bin/csh

############################## Initialize parameters ##############################
set subj = $1
set out = ~/ABA/local/results_offset/first_level/voxelwise_14TR_4plays_offset_reducedRuns/unassumed/ABA"$subj"
mkdir -p $out

if ($subj == 500 | $subj == 501 | $subj == 502 | $subj == 503 | $subj == 504 | $subj == 505 | $subj == 506) then
	set subj_name = QSQ"$subj"
else
	set subj_name = ABA"$subj"
endif

echo "Preprocessing Subject: $subj_name;"

set proj = ~/ABA/bswift
set local_proj = ~/ABA/local

set bad_runs = (`cat "$proj"/scripts/first_level/runs_to_exclude.txt | grep "$subj_name" | awk '{ print $2 }'`)
set num_of_bad_runs = $#bad_runs

echo "========================================================="
echo "Subject $subj_name has $num_of_bad_runs bad runs"
echo "========================================================="

set stimtimes = "$local_proj"/stimtimes/stimtimes_4_conds_Realigned_end_reducedRuns/"$subj_name"

if ($num_of_bad_runs > 0) then
    set input = "$proj"/dataset/preproc/ICA_AROMA/"$subj_name"/"$subj_name"_EP_Main_TR_MNI_2mm_SI_denoised_reducedRuns.nii.gz
    # set motion parameter files
    set fileRawMotion = "$proj"/dataset/preproc/ICA_AROMA/"$subj_name"/"$subj_name"_MotionPar_reducedRuns.txt'[0..5]'
    set fileDerMotion = "$proj"/dataset/preproc/ICA_AROMA/"$subj_name"/"$subj_name"_MotionParDeriv_reducedRuns.txt
    set fileCensMotion = "$proj"/dataset/preproc/ICA_AROMA/"$subj_name"/"$subj_name"_MotionCensor_1mm_reducedRuns.txt
else
    set input = "$proj"/dataset/preproc/ICA_AROMA/"$subj_name"/"$subj_name"_EP_TR_MNI_2mm_SI_denoised.nii.gz
    # set motion parameter files
    set fileRawMotion = "$proj"/dataset/preproc/"$subj_name"/func/"$subj_name"_MotionPar.txt'[1..6]'
    set fileDerMotion = "$proj"/dataset/preproc/"$subj_name"/func/"$subj_name"_MotionParDeriv.1D
    set fileCensMotion = "$proj"/dataset/preproc/"$subj_name"/func/"$subj_name"_MotionCensor_1mm.txt
endif

set play_basis = "CSPLIN(-11.25, 5, 14)"
echo $play_basis
set feed_basis = "GAM(8.6, 0.547, 1)"
echo $feed_basis
set rest_basis = "dmBLOCK(1)"

# multimask to use for blurring
set multiMask = "$local_proj"/templates/MNI152_T1_2mm_brain_GM_02182017.nii.gz

# set fileShockCensor15 = $out/"$subj_name"_ShockCensor15.txt
# set fileTouchCensor15 = $out/"$subj_name"_TouchCensor15.txt
# set fileCensMotionAndShock = $out/"$subj_name"_MotionAndShockCensor15.txt


############################## Setup concatenated list ##############################
# Get details of good runs
if ($subj == 600) then
   set total_runs = 6
   set run_len = '321.25 321.25 321.25 321.25 321.25 321.25'
   set nvolume = 257 #num of volumes per run
else if ($subj == 601) then
   set total_runs = 8
   set run_len = '416.25 416.25 416.25 416.25 416.25 416.25 416.25 416.25'
   set nvolume = 333 #num of volumes per run
else if ($subj == 605) then
   set total_runs = 4
   set run_len = '512.5 512.5 512.5 512.5'
   set nvolume = 410 #num of volumes per run
else
   set total_runs = 8
   set run_len = '512.5 512.5 512.5 512.5 512.5 512.5 512.5 512.5'
   set nvolume = 410 #num of volumes per run
endif

set good_runs = (`echo "tmp = '"$bad_runs"'.split(); bad_runs = [int(r) for r in tmp]; print(' '.join([str(i) for i in range(1,"$total_runs"+1) if i not in bad_runs]))" | python`)
set nruns = $#good_runs

# Create concat list
set concatList = ""
set i = 0
set idx = 0

while ( $i < $nruns )
  @ idx = $i * $nvolume
  set concatList = "$concatList"" ""$idx"
  @ i += 1
end
echo $concatList

rm -f $out/runConcatInfo.1D
echo $concatList > $out/runConcatInfo.1D

# # touch and shock censors
# rm -rf tmp1.txt
# timing_tool.py -timing "$stimtimes_raw"/regs/ABA"$subj"_feedback_Shocked_threat_high.txt -timing_to_1D tmp1.txt \
# 	-tr 1.25 -stim_dur 15.0 -min_frac 0.3 -timing_to_1D_warn_ok \
# 	-run_len $run_len

# 1deval -expr '1 - a' -a tmp1.txt >  $fileShockCensor15
# rm -rf tmp1.txt

# rm -rf tmp1.txt
# timing_tool.py -timing "$stimtimes_raw"/regs/ABA"$subj"_feedback_Touch_threat_low.txt -timing_to_1D tmp1.txt \
# 	-tr 1.25 -stim_dur 15.0 -min_frac 0.3 -timing_to_1D_warn_ok \
# 	-run_len $run_len

# 1deval -expr '1 - a' -a tmp1.txt >  $fileTouchCensor15
# rm -rf tmp1.txt

# # Shock, touch AND motion censor file
# rm -rf $fileCensMotionAndShock
# 1deval -overwrite -a "$fileShockCensor15" -b "$fileTouchCensor15" -c "$fileCensMotion" -expr 'a*b*c' > $fileCensMotionAndShock
	
	
echo "analysis of $subj from file: $input"
# cd $out

############################## Doing first stage regression (censor shock, motion, drifts) ##############################
3dDeconvolve -overwrite \
	-input $input \
	-noFDR \
    -mask $multiMask \
	-polort A \
	-local_times \
	-censor "$fileCensMotion" \
	-concat "$out"/runConcatInfo.1D \
	-num_stimts 13 \
	-stim_times 1 "$stimtimes"/regs/"$subj_name"_play_highT.txt "$play_basis" 		-stim_label 1 PLAY_highT \
    -stim_times 2 "$stimtimes"/regs/"$subj_name"_play_lowT.txt "$play_basis" 		-stim_label 2 PLAY_lowT \
    -stim_times 3 "$stimtimes"/regs/"$subj_name"_play_highR.txt "$play_basis" 		-stim_label 3 PLAY_highR \
    -stim_times 4 "$stimtimes"/regs/"$subj_name"_play_lowR.txt "$play_basis" 		-stim_label 4 PLAY_lowR \
  	-stim_times 5 "$stimtimes"/regs/"$subj_name"_feedback_Rec_highR.txt "$feed_basis" 	-stim_label 5 FEED_Rec_highR \
  	-stim_times 6 "$stimtimes"/regs/"$subj_name"_feedback_Rec_lowR.txt "$feed_basis" 	-stim_label 6 FEED_Rec_lowR \
  	-stim_times 7 "$stimtimes"/regs/"$subj_name"_feedback_NotRec_highT.txt "$feed_basis" 		-stim_label 7 FEED_NotRec_highT \
	-stim_times 8 "$stimtimes"/regs/"$subj_name"_feedback_NotRec_lowT.txt "$feed_basis" 		-stim_label 8 FEED_NotRec_lowT \
  	-stim_times 9 "$stimtimes"/regs/"$subj_name"_feedback_NotRec_highR.txt "$feed_basis" 		-stim_label 9 FEED_NotRec_highR \
  	-stim_times 10 "$stimtimes"/regs/"$subj_name"_feedback_NotRec_lowR.txt "$feed_basis"		-stim_label 10 FEED_NotRec_lowR \
	-stim_times 11 "$stimtimes"/regs/"$subj_name"_feedback_Rec_highT.txt "$feed_basis"		-stim_label 11 FEED_Rec_highT \
	-stim_times 12 "$stimtimes"/regs/"$subj_name"_feedback_Rec_lowT.txt "$feed_basis"			-stim_label 12 FEED_Rec_lowT \
    -stim_times_AM1 13 "$stimtimes"/regs/"$subj_name"_rest.txt "$rest_basis"			-stim_label 13 REST \
    -ortvec "$fileRawMotion" rawMotion \
    -ortvec "$fileDerMotion" derMotion \
	-x1D "$out"/"$subj_name"_Main_block_deconv.x1D \
	-x1D_uncensored "$out"/"$subj_name"_Main_block_deconv_uncensored.x1D \
	-errts "$out"/"$subj_name"_resids.nii.gz \
	-bucket "$out"/"$subj_name"_bucket.nii.gz \
	-cbucket "$out"/"$subj_name"_betas.nii.gz \
	-x1D_stop

# Calculate VIF using python script
chmod +x ~/ABA/local/scripts/getVIF.py
echo "==================== Calculating VIF ===================="
echo "$out"/"$subj_name"_Main_block_deconv.x1D | ~/ABA/local/scripts/getVIF.py 
echo "========================================================="

echo "***** Running 3dREMLfit *****"
	3dREMLfit -matrix "$out"/"$subj_name"_Main_block_deconv.x1D \
		-input $input \
		-overwrite \
		-mask $multiMask \
		-noFDR \
		-Rbeta "$out"/"$subj_name"_betas_REML.nii.gz \
		-Rbuck "$out"/"$subj_name"_bucket_REML.nii.gz \
		-Rvar "$out"/"$subj_name"_bucket_REMLvar.nii.gz \
 		-Rerrts "$out"/"$subj_name"_resids_REML.nii.gz \
		-Rwherr "$out"/"$subj_name"__wherrs_REML.nii.gz \
	
# 	1dcat "$out"/"$subj_name"_Main_block_deconv_bucket.1D > "$out"/"$subj_name"_Main_block_deconv_bucket_clean.1D

3dbucket -overwrite -prefix "$out"/"$subj_name"_PLAY_highT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[1..14]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_PLAY_lowT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[16..29]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_PLAY_highR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[31..44]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_PLAY_lowR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[46..59]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_Rec_highR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[61]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_Rec_lowR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[63]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_NotRec_highT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[65]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_NotRec_lowT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[67]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_NotRec_highR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[69]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_NotRec_lowR.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[71]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_Rec_highT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[73]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_FEED_Rec_lowT.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[75]'
3dbucket -overwrite -prefix "$out"/"$subj_name"_REST.nii.gz "$out"/"$subj_name"_bucket_REML.nii.gz'[77]'

echo "Exiting..."
exit




