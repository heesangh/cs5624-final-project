import json
import os

import pandas

JSON_FOLDER = 'json_responses'
COMBINED_DF_PATH = 'combined.parquet'


def main():
    json_paths = get_json_paths()

    dataframes = get_dataframes(json_paths)

    combined = pandas.concat(dataframes, ignore_index=True)

    combined.to_parquet(COMBINED_DF_PATH)


def get_json_paths():
    paths = []

    for root, dirs, files in os.walk(JSON_FOLDER):
        for file in files:
            paths.append(os.path.join(root, file))

    return paths


def get_dataframes(json_paths):
    dataframes = []

    for json_path in json_paths:
        try:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            dataframes.append(pandas.json_normalize(data))

        except Exception as e:
            print(f'Error reading {json_path}: {e}')

    return dataframes


if __name__ == '__main__':
    main()
