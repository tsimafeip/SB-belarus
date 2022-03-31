import os
from time import sleep

import requests
from bs4 import BeautifulSoup as bs
from typing import List, Optional
from tqdm import tqdm
import numpy as np

from sb_document import SbDocument

general_sb_prefix = 'https://www.sb.by'
page_with_links_prefix = r"https://www.sb.by/articles/main_policy/?PAGEN_2="
articles_prefix = 'https://www.sb.by/articles/'

https_prefix = 'https://www.sb.by'
http_prefix = 'http://sb.by'

PATH_TO_PAGE_LINKS_FILE = 'page_links.txt'


def get_soup(page_url: str):
    try:
        response = requests.get(page_url)
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    soup = bs(response.content, 'html.parser')

    sleep(np.random.randint(1, 5))

    return soup


def collect_links(start_page: int = 1, end_page: int = 968, path_to_file: str = PATH_TO_PAGE_LINKS_FILE) -> List[str]:
    page_links = []
    if os.path.isfile(path_to_file):
        with open(path_to_file, 'r', encoding='utf-8') as f:
            for line in f:
                page_links.append(line.strip())
        return page_links

    with open(path_to_file, 'w', encoding='utf-8') as links_file:
        for page_id in tqdm(range(end_page, start_page - 1, -1)):
            soup = get_soup(page_with_links_prefix + str(page_id))

            for link in soup.findAll('a', href=True):
                if link.attrs.get('class', None) == ['link-main']:
                    page_links.append(general_sb_prefix + link.attrs['href'] + '\n')
                    links_file.write(page_links[-1])

    return page_links


def parse_article_page(page_url: str, document_id: Optional[int] = None) -> SbDocument:
    soup = get_soup(page_url)

    text_link_pairs = [(item.text.strip(), item.attrs.get('href').replace(http_prefix, https_prefix))
                       for item in soup.find_all('a', attrs={'target': '_blank'})]
    similar_documents = [
        next_page_href
        for next_page_title, next_page_href in text_link_pairs
        if next_page_href.startswith(articles_prefix) and 'tags' not in next_page_href and next_page_title
    ]
    tags = soup.find('div', attrs={'class': "tags-list"})
    if tags:
        tags = tags.text.strip().split('\n')
    article_metainfo = soup.find('div', attrs={'class': "item-ajax article-accord"})
    if article_metainfo is None:
        article_metainfo = soup.find('div', attrs={'class': "item-ajax"})

    title_h1 = None
    if article_metainfo:
        title = article_metainfo.attrs['data-title'].strip()
        title_h1 = article_metainfo.attrs['data-title-h1'].strip()
        data_id = int(article_metainfo.attrs['data-id'].strip())
        # skip storing duplicate data
        if title_h1 == title:
            title_h1 = None
    else:
        title = soup.find('title').text.strip()
        data_id = document_id
    author = soup.find('meta', attrs={'property': 'article:author'}).attrs['content'].strip().lower()
    date = soup.find('meta', attrs={'property': 'article:modified_time'}).attrs['content'].strip()
    body = soup.find('div', attrs={'itemprop': 'articleBody'}).text.strip()

    # TODO: understand how to filter off blockquote staff better
    for doc_title in similar_documents:
        body = body.replace(doc_title, "")

    sb_document = SbDocument(title=title, publication_date=date, author=author, body=body, hyperlink=page_url,
                             title_h1=title_h1, tags=tags, document_id=data_id, similar_documents=similar_documents)

    return sb_document
