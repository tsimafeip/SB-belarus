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
    page_urls = collect_links(end_page=332, path_to_file='page_links_new.txt')

    print(len(page_urls))
