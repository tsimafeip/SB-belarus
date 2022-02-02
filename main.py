# mukovoz_url = 'https://www.sb.by/search/?q=муковозчик&PAGEN_2=28'
import os.path
from time import sleep

import requests
from bs4 import BeautifulSoup as bs
from typing import List, Tuple, Optional
import json
from tqdm import tqdm

general_sb_prefix = 'https://www.sb.by'
page_with_links_prefix = r"https://www.sb.by/articles/main_policy/?PAGEN_2="
articles_prefix = 'https://www.sb.by/articles/'

PATH_TO_PAGE_LINKS_FILE = 'page_links_2.txt'


class SbDocument:
    def __init__(self, document_id: int, title: str, date: str, author: str, body: str, hyperlink: str,
                 tags: Optional[List[str]] = None, title_h1: Optional[str] = None,
                 similar_documents: Optional[List[Tuple[str, str]]] = None):
        self.document_id = document_id
        self.title = title
        self.title_h1 = title_h1
        self.tags = tags
        self.similar_documents = similar_documents if similar_documents else []
        self.date = date
        self.author = author
        self.body = body
        self.hyperlink = hyperlink

    def dump_to_json(self):
        with open(os.path.join('../data', f'{self.document_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(self), f)

    @staticmethod
    def load_from_json(document_id: int) -> 'SbDocument':
        with open(os.path.join('../data', f'{document_id}.json'), 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            return SbDocument(**data_dict)


def get_soup(page_url: str):
    try:
        response = requests.get(page_url)
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    soup = bs(response.content, 'html.parser')

    sleep(5)

    return soup


def collect_links(start_page: int = 1, end_page: int = 959) -> List[str]:
    page_links = []
    if os.path.isfile(PATH_TO_PAGE_LINKS_FILE):
        with open(PATH_TO_PAGE_LINKS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                page_links.append(line.strip())
        return page_links

    for page_id in range(start_page, end_page):
        soup = get_soup(page_with_links_prefix + str(page_id))

        for link in soup.findAll('a', href=True):
            if link.attrs.get('class', None) == ['link-main']:
                page_links.append(general_sb_prefix + link.attrs['href'])

    with open(PATH_TO_PAGE_LINKS_FILE, 'w', encoding='utf-8') as links_file:
        for link in page_links:
            links_file.write(link + '\n')

    return page_links


def parse_article_page(page_url: str, document_id: int) -> SbDocument:
    path_to_file = os.path.join('../data', f'{document_id}.json')
    if os.path.isfile(path_to_file):
        return SbDocument.load_from_json(document_id)

    soup = get_soup(page_url)

    text_link_pairs = [(item.text, item.attrs.get('href')) for item in soup.find_all('a', attrs={'target': '_blank'})]
    similar_documents = [
        (next_page_title, next_page_href)
        for next_page_title, next_page_href in text_link_pairs
        if next_page_href.startswith(articles_prefix) and 'tags' not in next_page_href and next_page_title
    ]
    tags = soup.find('div', attrs={'class': "tags-list"})
    if tags:
        tags = tags.text.strip().split('\n')
    article_metinfo = soup.find('div', attrs={'class': "item-ajax article-accord"})
    title = title_h1 = None
    if article_metinfo:
        title = article_metinfo.attrs['data-title'].strip()
        title_h1 = article_metinfo.attrs['data-title-h1'].strip()
        # skip storing duplicate data
        if title_h1 == title:
            title_h1 = None
    else:
        title = soup.find('title').text.strip()
    author = soup.find('meta', attrs={'property': 'article:author'}).attrs['content'].strip()
    date = soup.find('meta', attrs={'property': 'article:modified_time'}).attrs['content'].strip()
    body = soup.find('div', attrs={'itemprop': 'articleBody'}).text.strip()

    sb_document = SbDocument(title=title, date=date, author=author,
                             body=body, document_id=document_id, hyperlink=page_url,
                             title_h1=title_h1, tags=tags, similar_documents=similar_documents)
    sb_document.dump_to_json()

    return sb_document


page_urls = collect_links(start_page=2, end_page=3)
sb_docs = []
for document_id, page_url in tqdm(enumerate(page_urls)):
    sb_document = parse_article_page(page_url, document_id=document_id + 20)
    sb_docs.append(sb_document)

t = 1
