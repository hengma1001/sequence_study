import os

hf_dir = "/homes/heng.ma/Research/md_pkgs/sequence_study/inference_scaling/cache"
os.environ["TRANSFORMERS_CACHE"] = hf_dir
from esmfold import run_inference

fasta_file = (
    "/lambda_stor/homes/heng.ma/Research/BVBRC/seqs/fig|2697049.1329549.CDS.17.fa"
)
# print(os.getenv("TRANSFORMERS_CACHE", "test"))
run_inference(fasta_file)
