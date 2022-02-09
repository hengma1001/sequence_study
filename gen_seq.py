# %%
import os 
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

wt_fasta = './ncd.fa'
wt_record = SeqIO.read(wt_fasta, "fasta")

print(wt_record.seq)
# %%
seq_savepath = 'seq_mut'
os.makedirs(seq_savepath, exist_ok=True)

mut_table = 'mutation'
with open(mut_table, 'r') as mut_p: 
    mut_list = []
    for line in mut_p: 
        if line.startswith('N:'): 
            mut_info = line.strip()[2:]
            source_res = mut_info[0]
            res_num = int(mut_info[1:-1]) - 1
            target_res = mut_info[-1]
            if seq_info[res_num] == source_res: 
                seq_info[res_num] = target_res
            else: 
                raise("Mutation source dosen't match the wild-type residue, "\
                    "check the mutation table...")
        elif line.startswith('M:'): 
            continue
        elif line.strip() == '': 
            record = SeqRecord(
                Seq(''.join(seq_info).replace('-', '')), 
                id=seq_title,
                name=seq_title,
                description="ncd mutation" 
            )
            file_save = '_'.join(seq_title.split())
            SeqIO.write(record, f'{seq_savepath}/ncd_{file_save}.fa', "fasta")
        else: 
            seq_title = line.strip()
            seq_info = list(wt_record.seq)
            print(seq_title)


# %%
