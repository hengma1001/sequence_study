# %%
from cProfile import run
import os
import glob
import subprocess

def run_shell(command, log=False):
    tsk = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True)
    tsk.wait()
    line = tsk.stdout.readline()
    if log: 
        log.write(line)
    else:
        print(line)


run_script = os.path.abspath('./run_alpha.sh')

seqs = glob.glob('./seq_mut/*.fa')
gpu_id = 1

fold_path = 'fold_mut'
os.makedirs(fold_path, exist_ok=True)

for seq in seqs: 
    seq_id = os.path.basename(seq)[:-3]
    print(seq_id)

    run_command = f'bash {run_script} -f {seq} -o {fold_path} -a {gpu_id}'
    run_log = f"{fold_path}/run_log"
    run_shell(run_command, open(run_log, 'wb'))


# %%
