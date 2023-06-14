import sys
import GPUtil
import logging
from argparse import ArgumentParser
from datetime import datetime
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, Optional

from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor
from proxystore.store import register_store
from proxystore.store.file import FileStore
from pydantic import root_validator

from parsl_config import ComputeSettingsTypes
from utils import BaseSettings, path_validator


def run_inference(
    fasta_file: Path,
    gpu_id: int,
    singularity_image: Path,
    exec_path: Path,
    data_dir: Path,
    conda_bin: Path,
    model_dir: Path,
    model_name: str,
    output_dir: Path,
) -> None:
    """Run an inference script on a test input_path and write an output csv
    to the output_dir with the same name as the input_path"""
    import shutil
    import tempfile
    import subprocess

    # created a temporary directory for openfold as it only reads dir
    temp_dir = tempfile.TemporaryDirectory() 
    temp_fasta = f"{temp_dir.name}/{os.path.basename(fasta_file)}"
    shutil.copy(fasta_file, temp_fasta)

    singularity_gpu_setup = f"SINGULARITYENV_CUDA_VISIBLE_DEVICES={gpu_id}"
    singularity_cmd = f"singularity run --nv --bind /lambda_stor/ {singularity_image}"
    command = (
        # f"{singularity_gpu_setup} "
        f"{singularity_cmd} "
        f"python {exec_path} {temp_dir.name} "
        f"{data_dir}/pdb_mmcif/mmcif_files "
        f"--output_dir {output_dir} "
        f"--uniref90_database_path {data_dir}/uniref90/uniref90.fasta "
        f"--mgnify_database_path {data_dir}/mgnify/mgy_clusters.fa "
        f"--pdb70_database_path {data_dir}/pdb70/pdb70 "
        f"--uniclust30_database_path {data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08 "
        f"--bfd_database_path {data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt "
        f"--jackhmmer_binary_path {conda_bin}/jackhmmer "
        f"--hhblits_binary_path {conda_bin}/hhblits "
        f"--hhsearch_binary_path {conda_bin}/hhsearch "
        f"--kalign_binary_path {conda_bin}/kalign "
        f"--model_device cuda:0 "
        f"--config_preset {model_name} "
        f"--jax_param_path {model_dir}/params/params_{model_name}.npz "
    )
    # Run the inference command
    subprocess.run(command.split())

    # Move from node local to persitent storage
    # if node_local_path is not None:
    #     shutil.move(output_path, output_dir / input_path.name)


class Thinker(BaseThinker):  # type: ignore[misc]
    def __init__(
        self, input_dir: Path, result_dir: Path, num_parallel_tasks: int, 
        maxLoad:float = 0.5, maxMemory:float = 0.5, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.task_idx = 0
        self.num_parallel_tasks = num_parallel_tasks
        self.input_files = list(input_dir.glob("*.fa"))[:10]
        self.gpu_ids = GPUtil.getAvailable(limit=self.num_parallel_tasks,
                    maxLoad=maxLoad, maxMemory=maxMemory)
        logging.info(f"Processing {len(self.input_files)} input files")

    def log_result(self, result: Result, topic: str) -> None:
        """Write a JSON result per line of the output file."""
        with open(self.result_dir / f"{topic}.json", "a") as f:
            print(result.json(exclude={"inputs", "value"}), file=f)

    def submit_task(self, topic: str, *inputs: Any) -> None:
        self.queues.send_inputs(
            *inputs, method=f"run_{topic}", topic=topic, keep_inputs=False
        )

    def submit_inference_task(self) -> None:
        # If we finished processing all the results, then stop
        if self.task_idx >= len(self.input_files):
            self.done.set()
            return

        input_file = self.input_files[self.task_idx]
        gpu_id = 0
        self.submit_task("inference", input_file, gpu_id)
        self.task_idx += 1

    @agent(startup=True)  # type: ignore[misc]
    def start_tasks(self) -> None:
        # Only submit num_parallel_tasks at a time
        for _ in range(self.num_parallel_tasks):
            self.submit_inference_task()

    @result_processor(topic="inference")  # type: ignore[misc]
    def process_inference_result(self, result: Result) -> None:
        """Handles the returned result of the inference function and log status."""
        self.log_result(result, "inference")
        if not result.success:
            logging.warning(f"Bad inference result: {result.json()}")

        # The old task is finished, start a new one
        self.submit_inference_task()


class WorkflowSettings(BaseSettings):
    """Provide a YAML interface to configure the workflow."""

    # Workflow setup parameters
    experiment_name: str = "experiment"
    """Name of the experiment to label the run directory."""
    runs_dir: Path = Path("runs")
    """Main directory to organize all experiment run directories."""
    run_dir: Path
    """Path this particular experiment writes to (set automatically)."""

    # Inference parameters
    singularity_image: Path
    """OpenFold singularity image"""
    exec_path: Path
    """Path to the python script implemeting the inference method."""
    fasta_dir: Path
    """Path to fasta files to run inference on."""
    data_dir: Path
    """Directory where all the OpenFold databases are stored."""
    conda_bin: Path
    """Singularity conda path for executables"""
    model_name: str
    """Selected model for openfold inference"""
    model_dir: Path
    """Pretrained model checkpoint to use for inference."""
    output_dir: Path
    """Directory to write csv output files to containing (SMILES, Database ID, docking score)."""
    
    # num_data_workers: int = 16
    # """Number of cores to use for datalaoder."""
    num_parallel_tasks: int = 6
    """Number of parallel task to run (should be the total number of GPUs)"""
    node_local_path: Optional[Path] = None
    """Node local storage option for writing output csv files."""

    compute_settings: ComputeSettingsTypes
    """The compute settings to use."""

    # validators
    # _input_dir_exists = path_validator("input_dir")
    # _script_path_exists = path_validator("script_path")
    # _checkpoint_dir_exists = path_validator("checkpoint_dir")

    def configure_logging(self) -> None:
        """Set up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.run_dir / "runtime.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    @root_validator(pre=True)
    def create_output_dirs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unique run path within run_dirs with a timestamp."""
        runs_dir = Path(values.get("runs_dir", "runs")).resolve()
        experiment_name = values.get("experiment_name", "experiment")
        timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
        run_dir = runs_dir / f"{experiment_name}-{timestamp}"
        run_dir.mkdir(exist_ok=False, parents=True)
        values["run_dir"] = run_dir
        # Specify task output directory
        values["output_dir"] = run_dir / "inference_output"
        values["output_dir"].mkdir(exist_ok=False, parents=True)
        return values


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    cfg = WorkflowSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.run_dir / "params.yaml")
    cfg.configure_logging()

    # Make the proxy store
    store = FileStore(name="file", store_dir=str(cfg.run_dir / "proxy-store"))
    register_store(store)

    # Make the queues
    queues = PipeQueues(
        serialization_method="pickle",
        topics=["inference"],
        proxystore_name="file",
        proxystore_threshold=10000,
    )

    # Define the parsl configuration (this can be done using the config_factory
    # for common use cases or by defining your own configuration.)
    parsl_config = cfg.compute_settings.config_factory(cfg.run_dir / "run-info")

    # Assign constant settings to each task function
    my_run_inference = partial(
        run_inference,
        singularity_image=cfg.singularity_image,
        exec_path=cfg.exec_path,
        data_dir=cfg.data_dir,
        conda_bin=cfg.conda_bin,
        model_dir=cfg.model_dir,
        model_name=cfg.model_name,
        output_dir=cfg.output_dir,
        )
    update_wrapper(my_run_inference, run_inference)

    doer = ParslTaskServer([my_run_inference], queues, parsl_config)

    thinker = Thinker(
        queue=queues,
        input_dir=cfg.fasta_dir,
        result_dir=cfg.run_dir / "result",
        num_parallel_tasks=cfg.num_parallel_tasks,
    )
    logging.info("Created the task server and task generator")

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info("Launched the servers")

        # Wait for the task generator to complete
        thinker.join()
        logging.info("Task generator has completed")
    finally:
        queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    # Clean up proxy store
    store.close()
