import os
from tqdm import tqdm

import sqlite3
import hashlib

from sb_scraper import collect_links, parse_article_page
from sb_document import SbDocument

'''
1) Download pages to sqllite
2) Create new table with preprocessed data
    - reset hyperlinks with document_ids
    - clean body
3) Data preprocessing
    - lemmatization
    - filter out stopwords
    

'''


def find_missed_links(con, table_name):
    db_links = []
    for row in con.cursor().execute(f'SELECT hyperlink FROM {table_name}'):
        db_links.append(row[0])

    with open('page_links.txt', 'r') as input_f, open('links_to_reload.txt', 'w') as output_f:
        all_links = {line.strip() for line in input_f.readlines()}
        diff = [line + '\n' for line in set(all_links) - set(db_links)]
        output_f.writelines(diff)


def hash_url(url):
    h = hashlib.sha1()
    h.update(url.encode())
    return h.hexdigest()


def create_db_table(con):
    # Create table
    con.cursor().execute('''CREATE TABLE docs
                    (document_id text, title text, title_h1 text, tags text, 
                    similar_documents text, publication_date datetime, author text,
                    body text, hyperlink text)''')


if __name__ == '__main__':
    # sample_document = SbDocument.load_from_json(0)

    # con = sqlite3.connect('sb_docs.db')

    # page_urls = collect_links()
    # sample_document = parse_article_page(page_urls[0], 0)
    # sample_document.dump_to_json()

    # sample_document.insert_row_to_sqllite(con)

    # create_db_table(con, sample_document)

    # parse_article_page('https://www.sb.by/articles/lukashenko-napravilsya-s-rabochim-vizitom-v-kitay-.html')

    DB_NAME = 'sb_articles.db'
    con = sqlite3.connect(DB_NAME)

    # con.cursor().execute(f'''create table if not exists documents
    #                     (document_id int PRIMARY KEY NOT NULL,
    #                     title text NOT NULL,
    #                     title_h1 text,
    #                     tags text,
    #                     similar_documents text,
    #                     publication_date datetime NOT NULL,
    #                     author text NOT NULL,
    #                     body text NOT NULL,
    #                     hyperlink text NOT NULL)''')

    # article = \
    #     parse_article_page('https://www.sb.by/articles/lukashenko-nazval-modernizatsiyu-mekhovogo-kombinata-v-vitebske-podarkom-narodu-k-7-noyabrya.html')
    # article.insert_row_to_sqllite(con, table_name='documents')

    find_missed_links(con, 'articles')

    con.close()

    # for row in con.cursor().execute('SELECT * FROM docs ORDER BY document_id'):
    #     print(row)

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    # con.close()
