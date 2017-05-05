import sys
import os
import urllib.request
import tarfile
import zipfile

def _print_download_progress(count, block_size, total_size):
    downloaded = count * block_size
    pct_complete = float(downloaded) / total_size
    msg = '\r- Download progress: {0:.1%} - {1}/{2}'.format(pct_complete, downloaded, total_size)

    sys.stdout.write(msg)
    sys.stdout.flush()

def maybe_download_and_extract(url, download_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if os.path.exists(file_path):
        print('Data has been already downloaded and unpacked')
        return

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    file_path, _ = urllib.request.urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
    print('\nDownload finished. Extracting files')

    if file_path.endswith('.zip'):
        zipfile.ZipFile(file=file_path, mode='r').extractall(download_dir)
    elif file_path.endswith(('.tar.gz', '.tgz')):
        tarfile.open(name=file_path, mode='r:gz').extractall(download_dir)

    print('Done')

