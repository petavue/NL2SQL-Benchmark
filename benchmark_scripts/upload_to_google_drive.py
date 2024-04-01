import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SERVICE_ACCOUNT_FILE = './service_account.json'
FOLDER_ID = input("Enter Google Drive folder link: ").strip()
FOLDER_ID = FOLDER_ID.split('/')[-1]

def authenticate():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def upload_folder_contents(folder_path, parent_folder_id):
    drive_service = authenticate()

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_metadata = {
                'name': item,
                'parents': [parent_folder_id]
            }
            media = MediaFileUpload(item_path, mimetype='application/octet-stream', resumable=True)

            try:
                file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                print(f'File uploaded successfully. File ID: {file.get("id")}')
            except Exception as e:
                print(f'An error occurred: {e}')
        elif os.path.isdir(item_path):
            folder_metadata = {
                'name': item,
                'parents': [parent_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            try:
                folder = drive_service.files().create(body=folder_metadata).execute()
                print(f'Folder created successfully. ID: {folder.get("id")}')
                upload_folder_contents(item_path, folder.get('id'))
            except Exception as e:
                print(f'An error occurred while creating folder: {e}')

if __name__ == "__main__":
    folder_path = input("Enter the path of the folder containing files you want to upload: ")
    upload_folder_contents(folder_path, FOLDER_ID)
