#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run each repeat as a separate job
    for SEED in {0..9}; do
        script="sbatch -J ${cfg_suffix}-${dataset} run/wrapper.sb ${main} --repeat 1 seed ${SEED} wandb.use True wandb.mode "offline" ${common_params}"
        echo $script
        eval $script
    done
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done


################################################################################
##### CSA
################################################################################

# Comment-out runs that you don't want to submit.
cfg_dir="configs/CSA"

DATASET="zinc"
run_repeats ${DATASET} CSA+RWSE "name_tag CSAwRWSE.10runs"
run_repeats ${DATASET} CSA+Rings "name_tag CSAwRWSE.10runs"


DATASET="pattern"
run_repeats ${DATASET} CSA "name_tag CSAwRWSE.lr0005.10run"


DATASET="cluster"
run_repeats ${DATASET} CSA "name_tag CSAwRWSE.lr0005.10run"

# DATASET="pcqm4m"  # NOTE: for PCQM4Mv2 we need SBATCH --mem=48G and up to 3days runtime; run only one repeat
# run_repeats ${DATASET} "CSADeep+SPDE-Rings-e150" "name_tag Deep_Rings_gelu_postpool_lr0.0001 train.auto_resume True train.ckpt_period 5"
# run_repeats ${DATASET} "CSA+RWSE-shared-e300" "name_tag gelu_postpool train.auto_resume True train.ckpt_period 5"

