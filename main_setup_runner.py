# main_runner.py
import sys
from pathlib import Path
# from docs.setup_scripts.generate_station_data import generate_catchment_data
# from book_docs.setup_scripts.precheck_utils import build_geometry_registry
from docs.setup_scripts.update_station_pages_and_intro import update_station_pages, generate_searchindex_js, generate_search_css
from docs.setup_scripts.introduction_page_utils import create_intro
from argparse import ArgumentParser

project_root = Path(__file__).resolve().parents[1]  # Adjust as needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

BASE_DIR = Path(__file__).resolve().parent

parser = ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="prod",
    help="Specify 'dev' for development mode or 'prod' for production mode.",
)
args = parser.parse_args()
site_url_prefix = "/" if args.mode == "dev" else "/distribution_estimation/"

station_data = update_station_pages(site_url_prefix=site_url_prefix)

# generate the introduction page
intro_template_path = BASE_DIR / 'docs' / "templates" / "intro_template.md"
output_path = BASE_DIR / 'docs' / "intro.md"
create_intro(intro_template_path, output_path, station_data)
generate_searchindex_js()
generate_search_css()

