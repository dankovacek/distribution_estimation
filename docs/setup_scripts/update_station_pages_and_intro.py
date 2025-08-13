import os, sys
import json
from pathlib import Path
from jinja2 import Template
from matplotlib.pylab import dtype
import pandas as pd

project_root = Path(__file__).resolve().parents[1]  # Adjust as needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from docs.setup_scripts.station_page_utils import process_static_station_page_html
# Get the folder this script is in
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

"""
Generate markdown pages for FDC estimation results by station/catchment..

This script creates individual markdown files for each station folder
and a main index page with a search bar for easy navigation.

Specify --mode flag to choose between 'dev' and 'prod' modes.
In 'prod' mode, the url for inserting the iframe includes 'distribution_estimation'
to work with the Jupyter book build process.  Running in dev mode allows 
you to test locally without the Jupyter book build process.
Usage:
    (production): python generate_station_pages.py --mode prod
    (development/testing): python generate_station_pages.py --mode dev
Testing: `python3 -m http.server --directory ../build/html 8000`
Production (to github pages): `ghp-import -n -p -f _build/html`

"""


def update_station_pages(site_url_prefix: str = ""):
    """Update or create README.md files for each station folder."""
    # station_data = scan_station_folders()
    # Process each station folder
    station_data_folder = BASE_DIR / '..' / "stations"
    # fpath = BASE_DIR / '..' / 'notebooks' / 'data' / 'BCUB_watershed_attributes_updated_20250227.csv'
    # stn_df = pd.read_csv(fpath, dtype={"official_id": str})
    hs_fpath = BASE_DIR / '..' / 'notebooks' / 'data' / 'HYSETS_watershed_properties.txt'
    hs_df = pd.read_csv(hs_fpath, dtype={"Official_ID": str}, sep=";")

    plots_folder = BASE_DIR / '..' / 'notebooks' / 'data' / 'results' / 'lstm_plots'
    plots_files = os.listdir(plots_folder)
    plot_stns = [f.split("_")[0] for f in plots_files if f.endswith(".html")]

    station_data = {}
    for official_id in plot_stns:
        stn_data = hs_df[hs_df["Official_ID"] == official_id].iloc[0].to_dict()
        # create a folder name based on the official id
        new_folder = station_data_folder / official_id
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        data = process_static_station_page_html(
            new_folder, official_id, site_url_prefix=site_url_prefix, stn_data=stn_data
        )
        station_data[official_id] = data
    return station_data


def generate_searchindex_js():
    static_dir = BASE_DIR / '..' / "_static/"
    os.makedirs(static_dir, exist_ok=True)
    search_js_path = static_dir / "searchindex.js"

    # Structured data snippet (update URL and search URL as needed)
    structured_data = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "FDC Estimation Results Archive",
        "alternateName": "FDC ERA",
        "url": "https://your-domain.com",  # Update with your URL
        "potentialAction": {
            "@type": "SearchAction",
            "target": {
                "@type": "EntryPoint",
                "urlTemplate": "https://your-domain.com/search.html?q={search_term_string}",
            },
            "query-input": "required name=search_term_string",
        },
    }

    js_content = (
        "var structuredData = " + json.dumps(structured_data, indent=2) + ";\n\n"
        "const SDscript = document.createElement('script');\n"
        "SDscript.setAttribute('type', 'application/ld+json');\n"
        "SDscript.textContent = JSON.stringify(structuredData);\n"
        "document.head.appendChild(SDscript);\n\n"
        "function filterStations() {\n"
        "  const input = document.getElementById('stationSearch').value.toLowerCase();\n"
        "  const results = document.getElementById('searchResults');\n"
        "\n"
        "  // Clear previous results\n"
        "  results.innerHTML = '';\n"
        "\n"
        "  if (input.length < 2) return;\n"
        "\n"
        "  // Filter stations\n"
        "  const matches = stations.filter(c =>\n"
        "    c.id.toLowerCase().includes(input) ||\n"
        "    c.name.toLowerCase().includes(input) ||\n"
        "    c.source.toLowerCase().includes(input) ||\n"
        "    `$${c.id}`.toLowerCase().includes(input)\n"
        "  );\n"
        "\n"
        "  // Display results (max 10)\n"
        "  const limitedMatches = matches.slice(0, 10);\n"
        "  limitedMatches.forEach(c => {\n"
        "    const div = document.createElement('div');\n"
        "    div.className = 'search-result';\n"
        "    div.innerHTML = `<a href='${c.folder}'>${c.id}: ${c.name}</a>`;\n"
        "    results.appendChild(div);\n"
        "  });\n"
        "\n"
        "  // Show message if too many results\n"
        "  if (matches.length > 10) {\n"
        "    const div = document.createElement('div');\n"
        "    div.className = 'search-more';\n"
        "    div.textContent = `... and ${matches.length - 10} more matches`;\n"
        "    results.appendChild(div);\n"
        "  }\n"
        "}\n"
    )

    with open(search_js_path, "w") as f:
        f.write(js_content)

    print(f"Generated search index JS at {search_js_path}")


def generate_search_css():
    static_dir = BASE_DIR / '..' / "_static/"
    os.makedirs(static_dir, exist_ok=True)
    css_path = static_dir / "custom.css"

    css_content = """
    .search-container {
      margin: 20px 0;
    }
    #stationSearch {
      width: 100%;
      padding: 10px;
      font-size: 16px;
      border: 1px solid #ddd;
      border-radius: 4px;
    }
    .search-results {
      max-height: 300px;
      overflow-y: auto;
      border: 1px solid #ddd;
      border-top: none;
      display: block;
    }
    .search-result {
      padding: 10px;
      border-bottom: 1px solid #eee;
    }
    .search-result:hover {
      background-color: #f5f5f5;
    }
    .search-more {
      padding: 10px;
      text-align: center;
      color: #777;
      font-style: italic;
    }
    """

    with open(css_path, "w") as f:
        f.write(css_content.strip())

    print(f"Generated search CSS at {css_path}")


# def update_summary_page_markdown(similarity_summary_table, revision_notes_table, plot_src, site_url_prefix=''):
#     # This function is a placeholder for any additional summary page updates
#         # HTML template
#     template_path = BASE_DIR / ".." / "templates" / "summary_template.md.j2"
#     with open(template_path, "r") as f:
#         md_template = Template(f.read())

#     rendered_md = md_template.render(
#         change_summary_table=similarity_summary_table,
#         revision_notes_table=revision_notes_table,
#         iframe_src=plot_src,
#     )

#     # Save markdown
#     md_path = BASE_DIR / ".." / 'revision_summary.md'
#     with open(md_path, "w", encoding="utf-8") as f:
#         f.write(rendered_md)
