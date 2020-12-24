import urllib
import gzip
import os
import urllib.request as req

user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
url = "https://data.gharchive.org/2020-10-31-{}.json.gz"
no_files = 2


def unzip_file(zip_file_name):
    unzipped_file_name = zip_file_name[: -3][4:] + ".json"
    with gzip.open(zip_file_name, 'rb') as f_in:
        with open(unzipped_file_name, 'wb') as f_out:
            zip_file_content = f_in.read()
            f_out.write(zip_file_content)


for counter in range(no_files):
    headers = {'User-Agent': user_agent, }
    request_url = url.format(counter)
    # The assembled request
    request = req.Request(request_url, None, headers)
    response = req.urlopen(request)
    # The data u need
    data = response.read()
    file_name = "gha_2020_10_31_{}.gz".format(counter)
    f = open(file_name, 'wb')
    f.write(data)
    f.close()
    unzip_file(file_name)
    os.remove(file_name)