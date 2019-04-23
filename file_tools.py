import os
import sys


def assert_file_exists(file):
    if not os.path.isfile(file):
        print(f'File: {file} does not exist.')
        sys.exit(1)
