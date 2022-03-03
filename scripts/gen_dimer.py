import os 
import glob
from Bio import SeqIO

seqs = glob.glob('./seq_mut/*.fa')

save_path = 'seq_dimers'
os.makedirs(save_path, exist_ok=True)
for seq in seqs: 
    record = SeqIO.read(seq, "fasta")
    dimer = [record] * 2

    seq_label = os.path.basename(seq)[:-3]
    SeqIO.write(dimer, f'{save_path}/{seq_label}_dimer.fa', "fasta")
