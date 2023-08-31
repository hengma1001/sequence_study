import os

import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

# os.environ["TRANSFORMERS_CACHE"] = "./cache/huggingface/"

torch.backends.cuda.matmul.allow_tf32 = True


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_inference(fasta_file):
    run_label = os.path.basename(fasta_file)[:-3]

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )

    model = model.cuda()
    # Uncomment to switch the stem to float16
    model.esm = model.esm.half()
    # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
    model.trunk.set_chunk_size(64)

    # This is the sequence for human GNAT1, because I worked on it when
    # I was a postdoc and so everyone else has to learn to appreciate it too.
    # Feel free to substitute your own peptides of interest
    # Depending on memory constraints you may wish to use shorter sequences.
    record = SeqIO.read(fasta_file, "fasta")
    test_protein = str(record.seq)

    tokenized_input = tokenizer(
        [test_protein], return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    tokenized_input = tokenized_input.cuda()

    with torch.no_grad():
        output = model(tokenized_input)

    pdb = convert_outputs_to_pdb(output)

    with open(f"{run_label}.pdb", "w") as f:
        f.write("".join(pdb))
