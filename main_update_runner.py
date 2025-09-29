# main_runner.py
import os, sys
from pathlib import Path
from book_docs.setup_scripts.precheck_utils import build_geometry_registry
from integration import run_integration_workflow, run_update_workflow, run_change_summary_workflow

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="prod",
    help="Specify 'dev' for development mode or 'prod' for production mode.",
)
args = parser.parse_args()
site_url_prefix = "" if args.mode == "dev" else "/camel_farrier"

project_root = Path(__file__).resolve().parents[1]  # Adjust as needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

BASE_DIR = Path(__file__).resolve().parent

config = load_file_config()
files = find_geometry_files()

existing_folders = os.listdir(BASE_DIR / "book_docs" / "catchments")
existing_sites = [
    f.split("-")[1] for f in existing_folders
]  # Extract the official IDs from the folder names
existing_site_files = [
    file for file in files if str(file.resolve()).split("/")[-1].split('_')[0] in existing_sites
]

registry, missing_cols = build_geometry_registry(existing_site_files, config)

output_dir = BASE_DIR / "book_docs" / "catchments"
run_integration_workflow(registry, config, output_dir)
run_update_workflow(site_url_prefix=site_url_prefix)
run_change_summary_workflow(site_url_prefix=site_url_prefix)


