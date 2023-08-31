import logging
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partial, update_wrapper
from pathlib import Path
from typing import Any, Dict, Optional

from colmena.models import Result
from colmena.queue.python import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, agent, result_processor

# from esmfold import run_inference
from parsl_config import ComputeSettingsTypes
from proxystore.store import register_store
from proxystore.store.file import FileStore
from pydantic import root_validator
from utils import BaseSettings


def run_inference(fasta_file, output_dir, hf_dir):
    import os

    os.environ["TRANSFORMERS_CACHE"] = str(hf_dir)

    import torch
    from Bio import SeqIO
    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
    from transformers.models.esm.openfold_utils.protein import to_pdb

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
                chain_index=outputs["chain_index"][i]
                if "chain_index" in outputs
                else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs

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

    with open(f"{output_dir}/{run_label}.pdb", "w") as f:
        f.write("".join(pdb))


class Thinker(BaseThinker):  # type: ignore[misc]
    def __init__(
        self,
        input_dir: Path,
        result_dir: Path,
        num_parallel_tasks: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        result_dir.mkdir(exist_ok=True)
        self.result_dir = result_dir
        self.task_idx = 0
        self.num_parallel_tasks = num_parallel_tasks
        self.input_files = list(input_dir.glob("*.fa"))
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
        self.submit_task("inference", input_file)
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
    fasta_dir: Path
    """Path to fasta files to run inference on."""
    hf_dir: Path
    """hugging face cache dir"""
    output_dir: Path
    """output path (set automatically)"""

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
        run_inference, output_dir=cfg.output_dir, hf_dir=cfg.hf_dir
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
