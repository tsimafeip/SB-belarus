import boto3
import hashlib
import gzip
import requests
import os
import redis
from bs4 import BeautifulSoup as bs
from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

load_dotenv(verbose=True)

AWS_DIRECTORY = os.getenv('AWS_BUCKET_DIR')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET')
BASE_URL = os.getenv('BASE_URL')

app = Celery('crawler', broker=f'{os.getenv("RABBIT_URL")}', backend='rpc://')
logger = get_task_logger(__name__)
session = boto3.session.Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.getenv('AWS_ACCESS_KEY_SECRET'),
                                region_name=os.getenv('AWS_REGION'))
s3_resource = session.resource('s3')
redis_resource = redis.Redis(host=os.getenv('REDIS_IP'))


class BadResponse(Exception):
    def __init__(self, url, code):
        self.url = url
        self.code = code


def upload_file_to_bucket(bucket_name, file_path, directory):
    file_dir, file_name = os.path.split(file_path)

    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=file_path,
        Key=f'{directory}/{file_name}',
        ExtraArgs={'ACL': 'public-read'}
    )


def hash_url(url):
    h = hashlib.sha1()
    h.update(url.encode())
    return h.hexdigest()


uploaded_pages_cache = set([obj['Key'].split('/')[-1] for obj in session.client('s3').list_objects_v2(
    Bucket=AWS_BUCKET_NAME, Prefix=AWS_DIRECTORY, MaxKeys=50000).get('Contents', [{'Key': '/'}])])


def page_exists_in_s3(url):
    if url in uploaded_pages_cache:
        return True

    obj = s3_resource.Object(AWS_BUCKET_NAME, f'{AWS_DIRECTORY}/{hash_url(url)}.html.gz')
    try:
        obj.load()
        uploaded_pages_cache.add(url)
        return True
    except:
        return False


def is_page_processed(url):
    return redis_resource.get(url) is not None


@app.task()
def fetch_url(url):
    if page_exists_in_s3(url):
        return

    logger.info(url)
    response = requests.get(url)
    if response.status_code != 200 and response.status_code != 404:
        logger.error(f'{response.status_code}-{url}')
        raise BadResponse(url, response.status_code)

    parsed = bs(response.text, 'lxml')
    content = parsed.find('div', {'id': 'content'})

    url_hash = hash_url(url)
    filepath = f'files/{url_hash}.html.gz'
    with gzip.open(filepath, 'wb') as f:
        f.write(response.text.encode())
    upload_file_to_bucket(AWS_BUCKET_NAME, filepath, AWS_DIRECTORY)
    os.remove(filepath)

    for link in content("a"):
        if 'href' in link.attrs:
            href = link.attrs['href']
            if not href.startswith('/wiki/'):
                continue
            if ':' in href and ':_' not in href:
                continue

            full_link = BASE_URL + href
            if not is_page_processed(full_link):
                fetch_url.delay(full_link)
                redis_resource.set(full_link, hash_url(full_link))
                logger.warning(f'New url: {href}')


if __name__ == '__main__':
    fetch_url.delay(BASE_URL)
