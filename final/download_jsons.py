import concurrent.futures
import json
import os

import requests

API_URL = f'https://crashviewer.nhtsa.dot.gov/CISS/CISSCrashData/'
SAVE_FOLDER = 'json_responses'
MAX_CONCURRENT_DOWNLOADS = 10


def main():
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    with open('case_ids.txt', 'r') as case_ids_file:
        case_ids = case_ids_file.read().splitlines()

    case_ids = [int(case_id) for case_id in case_ids]

    download_jsons(case_ids)


def download_jsons(case_ids):
    progress_tracker = {
        'completed': 0,
        'total': len(case_ids)
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        futures = {executor.submit(single_download, case_id, progress_tracker): case_id for case_id in case_ids}

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except:
                pass


def single_download(case_id, progress_tracker):
    params = {'crashid': case_id}

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()

        data = response.json()

        file_path = os.path.join(SAVE_FOLDER, f'{case_id}.json')
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        progress_tracker['completed'] += 1
        print_progress(progress_tracker['completed'], progress_tracker['total'])

    except Exception as e:
        progress_tracker['completed'] += 1
        print_progress(progress_tracker['completed'], progress_tracker['total'])
        print(f'Failed to fetch/save data for case {case_id}: {e}')


def print_progress(completed, total):
    percent_complete = (completed / total) * 100
    print(f'\rProgress: {completed}/{total} ({percent_complete:.2f}%)', end='', flush=True)


if __name__ == '__main__':
    main()
