import boto3
import os
import json

S3_ENDPOINT_URL='https://storage.yandexcloud.net'
S3_BUCKET_NAME='vkr2'

with open('keys.json', 'r', encoding='utf-8') as file:
    keys = json.load(file)

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=keys['key'],
    aws_secret_access_key=keys['secret_key']
)

file_name = "fire_0001.jpg"

def upload_file_to_s3(file_path, bucket_name):
    """Загружает файл в S3 и возвращает статус"""
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": "Файл не найден"}

        s3.upload_file(
            file_path,
            bucket_name,
            file_name,
        )

        file_url = f"{S3_ENDPOINT_URL}/{bucket_name}/{file_name}"
        
        return {
            "status": "success",
            "message": "Файл успешно загружен",
            "url": file_url
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка загрузки: {str(e)}"
        }


result = upload_file_to_s3(file_name, S3_BUCKET_NAME)
print(result)
