"""Utilities tests"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized, param

from uetai.utils import download_from_url, download_from_google_drive


class TestDownloadFunction(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDownloadFunction, self).__init__(*args, **kwargs)
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.tmp_dir.mkdir(parents=True, exist_ok=True)  # create tmp dir

    @parameterized.expand([
        param(
            'https://file-examples-com.github.io/uploads/2017/10/file-sample_100kB.odt',
            'file-sample_100kB.odt'
        ),
        param(
            'https://file-examples-com.github.io/uploads/2017/02/zip_5MB.zip',
            'zip_5MB.zip'
        )
    ])
    def test_download_from_url(self, url, filename):
        download_from_url(url=url, save_dir=self.tmp_dir)
        download_path = Path(os.path.join(self.tmp_dir, filename))
        self.assertTrue(download_path.exists())

    @parameterized.expand([
        param('1Sm66yNL5GeKIQIf9F2nGGcywZaIs7CZq'),
        param('1nrYxyYVwMsRFBF0tSr-JCD1lJQJ9gVMv'),
        param('https://drive.google.com/uc?id=1Sm66yNL5GeKIQIf9F2nGGcywZaIs7CZq'),
    ])
    def test_download_from_google_drive(self, id_or_url):
        save_path = os.path.join(self.tmp_dir, 'gdrive')
        save_path = download_from_google_drive(id_or_url, save_path=save_path)
        download_path = Path(os.path.join(save_path, 'text.txt'))
        self.assertTrue(download_path.exists())
        os.remove(download_path)  # remove file for next test


if __name__ == '__main__':
    unittest.main()
