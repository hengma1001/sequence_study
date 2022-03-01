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


run_script = '/lambda_stor/homes/heng.ma/Research/rosetta_fold/RoseTTAFold/run_e2e_ver.sh'

seqs = glob.glob('./seq_mut/*.fa')

fold_path = 'fold_mut'
os.makedirs(fold_path, exist_ok=True)

for seq in seqs: 
    seq_id = os.path.basename(seq)[:-3]
    print(seq_id)
    
    run_path = f'{fold_path}/{seq_id}'
    os.makedirs(run_path, exist_ok=True)

    run_command = f'{run_script} {seq} {run_path}'
    run_log = f"{run_path}/run_log"
    run_shell(run_command, open(run_log, 'wb'))


# %%