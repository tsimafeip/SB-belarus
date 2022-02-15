import os
from tqdm import tqdm

import sqlite3

from sb_scraper import collect_links, parse_article_page


'''
1) Download pages to sqllite
2) Create new table with preprocessed data
    - reset hyperlinks with document_ids
    - clean body
3) Data preprocessing
    - lemmatization
    - filter out stopwords
    

'''

if __name__ == '__main__':
    page_urls = collect_links()

    con = sqlite3.connect('example.db')

    sb_docs = []
    with open(os.path.join('data', 'sb_docs.json'), 'w', encoding='utf-8') as f:
        for document_id, page_url in tqdm(enumerate(page_urls)):
            sb_document = parse_article_page(page_url, document_id)
            sb_docs.append(sb_document)
