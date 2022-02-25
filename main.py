import datetime
from collections import Counter

from tqdm import tqdm

import sqlite3

from helper import load_data_from_db, count_most_common_tags, preprocess_text, preprocess_docs, read_preprocessed_docs
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
    sb_documents = load_data_from_db(DB_NAME, TABLE_NAME)

    #words = preprocess_docs(sb_documents)
    docs, words = read_preprocessed_docs()
    # most popular words
    words_counter = Counter(words)
    print(words_counter.most_common(20))

    # most popular tags
    count_most_common_tags(sb_documents)

    # articles per month
    

    # num of unique tokens
    print('Num of tokens: ', len(words))
    print('Num of unique tokens: ', len(words_counter))

    t = 1
