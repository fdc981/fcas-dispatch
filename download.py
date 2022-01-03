import requests
import pathlib
import bs4
import re
import shutil
import os

response = requests.get("https://www.nemweb.com.au/REPORTS/CURRENT/Daily_Reports/")
soup = bs4.BeautifulSoup(response.text)
zip_links = soup.find_all('a', string=re.compile('.zip'))

# path string of the output folder
out_path = "data/"

# Download all zip files
pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

for a_tag in zip_links:
    filename = a_tag.attrs['href'].split('/')[-1]

    if not pathlib.Path("data/" + filename.replace("zip", "CSV")).exists():
        url = "https://www.nemweb.com.au" + a_tag.attrs['href']
        response = requests.get(url)

        with open(f"{out_path}/{filename}", 'wb') as f:
            f.write(response.content)

# Extract and remove each zip file
zip_paths = pathlib.Path(out_path).glob('*.zip')

for path in zip_paths:
    shutil.unpack_archive(str(path), str(path.parent))
    os.remove(str(path))
