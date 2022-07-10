from pathlib import Path
import random
import time
from urllib.request import urlopen

import click
import requests

def scrape_midi_data(dest_dir: Path):
    years = [2002, 2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
    base_path = "https://piano-e-competition.com"

    def _fix_midi_url(url: Path) -> Path:
        parts = url.parts
        if len(parts) == 0 or str(parts[0]).lower() in ["midifiles", "contestantbios"]:
            return url
        return _fix_midi_url(Path(*parts[1:]))

    for year in years:
        index_path = f"{base_path}/midi_{year}.asp"
        click.echo(f"midi_scraper fetching {year} index at {index_path}")
        html_str = str(urlopen(index_path).read())
        midi_urls = [Path(url) for url in html_str.split("\"") if ".mid" in url.lower()]

        for midi_url in midi_urls:
            midi_url = _fix_midi_url(midi_url)
            full_url = base_path + "/" + str(midi_url)
            filename = str(year) + "_" + str(Path(full_url).name)
            out_path = dest_dir / filename

            if out_path.exists():
                click.echo(f"midi_scraper skipping {out_path} as it already exists")
                continue

            click.echo(f"midi_scraper downloading {full_url} and saving to {out_path}")
            response = requests.get(full_url)
            with open(out_path, "wb") as f:
                f.write(response.content)

            time.sleep(random.uniform(0, 3)) # don't hammer their servers
