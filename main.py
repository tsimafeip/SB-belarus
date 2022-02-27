import datetime
from collections import Counter, defaultdict

from tqdm import tqdm

import sqlite3

from helper import load_data_from_db, count_most_common_tags, preprocess_text, preprocess_docs, read_preprocessed_docs, \
    count_articles_by_month, run_preliminary_analysis
from sb_scraper import collect_links, parse_article_page

DB_NAME = 'sb_articles.db'
TABLE_NAME = 'documents'

'''
select count(distinct hyperlink) from articles

1) Download pages to sqllite
2) Create new table with preprocessed data
    - reset hyperlinks with document_ids
    - clean body
3) Data preprocessing
    - lemmatization
    - filter out stopwords
'''

if __name__ == '__main__':
    sb_documents = sorted(load_data_from_db(DB_NAME, TABLE_NAME), key=lambda x: x.publication_date)
    run_preliminary_analysis(sb_documents)
