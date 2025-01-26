import os
import requests
from urllib.parse import urlencode


def download_file(public_link, save_path=None):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    final_url = base_url + urlencode(dict(public_key=public_link))
    response = requests.get(final_url)
    parse_href = response.json()['href']

    url = parse_href
    start_filename = url.find('filename=')
    end_filename = url[start_filename:].find('&')
    end_name = start_filename + end_filename
    download_url = requests.get(url)

    if not save_path:
        filename = url[start_filename:end_name][9:]
        save_path = os.path.join(os.getcwd(), filename)
    with open(save_path, 'wb') as ff:
        ff.write(download_url.content)