import os

from esmfold import run_inference

os.environ["TRANSFORMERS_CACHE"] = "./cache/huggingface/"

fasta_file = (
    "/lambda_stor/homes/heng.ma/Research/BVBRC/seqs/fig|2697049.1329549.CDS.17.fa"
)
run_inference(fasta_file)
