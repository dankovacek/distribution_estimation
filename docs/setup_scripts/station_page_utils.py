import os
import numpy as np
import geopandas as gpd
import pandas as pd


from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from jinja2 import Template
from bokeh.palettes import Bokeh6, Category10

# Get the folder this script is in
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import xyzservices.providers as xyz

tiles = xyz.OpenStreetMap.Mapnik


def render_html_from_plot(p, html_path):
    # Generate the components
    script, div = components(p)
    plot_template = Template(
        """
        {{ resources }}
        {{ script }}
        {{ div | safe }}
        """
    )

    # plot_template = Template(
        # """
    # <!DOCTYPE html>
    # <html lang="en">
    # <head>
    #     <meta charset="UTF-8">
    #     {{ resources }}
    #     {{ script }}
    #     <style> body { margin: 0; } </style>
    # </head>
    # <body>
    #     {{ div | safe }}
    # </body>
    # </html>
    # """
    # )
    
    rendered_plot_html = plot_template.render(
        script=script, div=div, resources=CDN.render()
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(rendered_plot_html)


def get_station_folders():
    """
    Get all station folders in the stations directory.
    """
    base_folder = BASE_DIR / ".." / "stations"
    station_folders = [
        folder
        for folder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, folder))
    ]
    return station_folders


def process_static_station_page_html(
    folder: str,
    official_id: str ="",
    site_url_prefix: str = "/",
    stn_data: dict = {},
):

    station_name = stn_data.get("Name", "Unknown Station")
    source_code = stn_data.get("Source", "Unknown Source")

    # HTML template
    template_path = BASE_DIR / ".." / "templates" / "station_page_template.md.j2"
    with open(template_path, "r") as f:
        md_template = Template(f.read())

    iframe_src = (
        f"{site_url_prefix}_static/stations/{official_id}_fdc.html"
    )

    rendered_md = md_template.render(
        source_code=source_code,
        official_id=official_id,
        station_name=station_name,
        station_title=f"{official_id}: {station_name}",
        iframe_src=iframe_src,
    )

    # Save markdown
    md_path = folder / f"{official_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(rendered_md)

    print(f"Generated Markdown with iframe at {md_path}")

    return {
        "source_code": source_code,
        "official_id": official_id,
        "name": station_name,
        "folder": folder,
    }
