import os
import sys


# set global variables
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
interactive = os.isatty(sys.stdout.fileno())
hpc_storage_dir = "/rds/project/rds-eWkDxBhxBrQ"

# default directory for writing to (checkpoints, data cache, plots, etc)
if on_cluster:
    output_dir = os.path.join(hpc_storage_dir, "lauro/meta-models/outputs")
else:
    output_dir = os.path.join(module_path, "outputs")

os.makedirs(output_dir, exist_ok=True)
