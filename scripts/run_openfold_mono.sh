#!/bin/bash 

usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-o <output_dir>   Path to a directory that will store the results."
        echo "-f <fasta_path>   Path to a FASTA file containing one sequence"
        echo "-a <gpu_devices>  Comma separated list of devices to pass to 'CUDA_VISIBLE_DEVICES' (default: 0)"
        echo ""
        exit 1
}

while getopts ":o:f:a:g" i; do
        case "${i}" in
        o)
                output_dir=$OPTARG
        ;;
        f)
                fasta_path=$OPTARG
        ;;
        a)
                gpu_devices=$OPTARG
        ;;
        esac
done

if [[ "$fasta_path" == "" || "$output_dir" == "" ]]
then
    usage
fi

if [[ "$use_gpu" == "" ]]
then
    use_gpu=true
fi

fasta_path=`realpath -s $fasta_path`
output_dir=`realpath -s $output_dir`
# GPU_ID=$3 if $3 else 0

# image location and commands
SIMG="/lambda_stor/homes/heng.ma/Research/md_pkgs/openfold/openfold.sif"
SIMG_GPU="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$gpu_devices"
SINGULARITY="singularity run --nv --bind /lambda_stor/ $SIMG"
conda_bin="/opt/conda/bin/"

data_dir="/lambda_stor/data/hsyoo/AlphaFoldData"
new_data_dir="/lambda_stor/homes/heng.ma/Research/alpha_fold/dbs"

echo $SINGULARITY
echo $output_dir
echo $SIMG_GPU
# exit 

export $SIMG_GPU
$SINGULARITY \
python /opt/openfold/run_pretrained_openfold.py \
    $fasta_path \
    $data_dir/pdb_mmcif/mmcif_files \
    --uniref90_database_path $data_dir/uniref90/uniref90.fasta \
    --mgnify_database_path $data_dir/mgnify/mgy_clusters.fa \
    --pdb70_database_path $data_dir/pdb70/pdb70 \
    --uniclust30_database_path $data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --output_dir $output_dir \
    --bfd_database_path $data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path $conda_bin/jackhmmer \
    --hhblits_binary_path $conda_bin/hhblits \
    --hhsearch_binary_path $conda_bin/hhsearch \
    --kalign_binary_path $conda_bin/kalign \
    --model_device "cuda:0" \
    --config_preset "model_1_ptm" \
    --jax_param_path $new_data_dir/params/params_model_1_ptm.npz
    

#     --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt


# --bfd_database_path=$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt  \
# --mgnify_database_path=$data_dir/mgnify/mgy_clusters.fa  \
# --pdb70_database_path=$data_dir/pdb70/pdb70  \
# --template_mmcif_dir=$data_dir/pdb_mmcif/mmcif_files  \
# --obsolete_pdbs_path=$data_dir/pdb_mmcif/obsolete.dat  \
# --uniclust30_database_path=$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08  \
# --uniref90_database_path=$data_dir/uniref90/uniref90.fasta  \
# --data_dir=$new_data_dir  \
# --output_dir=$output_dir \
# --fasta_paths=$fasta_path  \
# --max_template_date=2020-05-01  \
# --db_preset=full_dbs \
# --model_preset=monomer \
# --run_relax=true \
# --use_gpu_relax=true \
# --benchmark=false  \
# --logtostderr



# echo $SINGULARITY
# echo $output_dir
# SINGULARITYENV_CUDA_VISIBLE_DEVICES=7 
# singularity run --nv -B /lambda_stor/  
# /lambda_stor/data/hsyoo/AlphaFoldImage/alphafold.sif 
# /opt/miniconda3/envs/alphafold/bin/python 
# /opt/alphafold/run_alphafold.py 
# --hhblits_binary_path=/opt/miniconda3/envs/alphafold/bin/hhblits 
# --hhsearch_binary_path=/opt/miniconda3/envs/alphafold/bin/hhsearch 
# --jackhmmer_binary_path=/opt/miniconda3/envs/alphafold/bin/jackhmmer 
# --kalign_binary_path=/opt/miniconda3/envs/alphafold/bin/kalign 
# --bfd_database_path=/lambda_stor/data/hsyoo/AlphaFoldData/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt 
# --mgnify_database_path=/lambda_stor/data/hsyoo/AlphaFoldData/mgnify/mgy_clusters.fa 
# --template_mmcif_dir=/lambda_stor/data/hsyoo/AlphaFoldData/pdb_mmcif/mmcif_files 
# --obsolete_pdbs_path=/lambda_stor/data/hsyoo/AlphaFoldData/pdb_mmcif/obsolete.dat 
# --pdb70_database_path=/lambda_stor/data/hsyoo/AlphaFoldData/pdb70/pdb70 
# --uniclust30_database_path=/lambda_stor/data/hsyoo/AlphaFoldData/uniclust30/uniclust30_2018_08/uniclust30_2018_08 
# --uniref90_database_path=/lambda_stor/data/hsyoo/AlphaFoldData/uniref90/uniref90.fasta 
# --data_dir=/lambda_stor/data/hsyoo/AlphaFoldData 
# --output_dir=/homes/heng.ma/Research/Fab/seq_study/test/ 
# --fasta_paths=/homes/heng.ma/Research/Fab/seq_study/ncd.fa 
# --model_names=model_1,model_2,model_3,model_4,model_5 
# --max_template_date=2020-05-01 
# --preset=casp14 
# --benchmark=false 
# --logtostderr
