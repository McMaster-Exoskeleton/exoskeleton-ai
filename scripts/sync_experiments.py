import torch
import argparse
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

SPREADSHEET_ID = "1-Si8tZm_vpvJ6EqJAHpN2B6QHZuvhQBis1RzGHjbRI4"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def get_service_client(service_account_file):
    # Load credentials directly from the file
    creds = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES
    )
    return build('sheets', 'v4', credentials=creds)


def sheet_get_keys(service):

    # get keys in first column ("A")
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range="Sheet1!A:A"
    ).execute()

    values = result.get('values', [])

    if not values:
        return []

    return [row[0] for row in values if row]


def sheet_append_experiments(service, data):

    body = {'values': data}

    # Append data to Sheet1 starting at A1
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range="Sheet1!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()
    print("Update successful!")


def get_experiments_rows(experiment_path, name):
    base_path = Path(experiment_path)
    rows = []

    for folder_path in base_path.glob("*/*/"):

        # Define the file paths
        results_file = folder_path / 'training_results.pt'
        config_file = folder_path / 'config.yaml'

        if not results_file.exists() or not config_file.exists():
            continue

        date_key = folder_path.parts[-2]  # date
        time_key = folder_path.parts[-1]  # time
        full_key = f"{name}/{date_key}/{time_key}"

        rows.append([full_key,
                     json.dumps(torch.load(
                         results_file, weights_only=True), indent=4),
                     config_file.read_text()
                     ])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync experiment results to Google Sheets.")

    # Required positional argument
    parser.add_argument(
        "name",
        type=str,
        help="Your name (cannot be null)."
    )

    # Optional arguments with defaults from your code
    parser.add_argument(
        "--service_account",
        type=str,
        default='service_account.json',
        help="Path to the Google service account JSON file."
    )

    parser.add_argument(
        "--experiment_path",
        type=str,
        default='outputs',
        help="Path to the outputs folder containing experiment data."
    )

    args = parser.parse_args()

    # Update global variables or local logic based on CLI input
    SERVICE_ACCOUNT_FILE = args.service_account
    NAME = args.name
    EXPERIMENT_PATH = args.experiment_path

    service = get_service_client(SERVICE_ACCOUNT_FILE)
    existing_keys = set(sheet_get_keys(service))
    rows = [row for row in get_experiments_rows(
        EXPERIMENT_PATH, NAME) if row[0] not in existing_keys]
    sheet_append_experiments(service, rows)
