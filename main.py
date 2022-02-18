from tqdm import tqdm

import sqlite3

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
    page_urls = collect_links(path_to_file='links_to_reload.txt')

    con = sqlite3.connect(DB_NAME)

    # create table if it does not exist
    con.cursor().execute(f'''create table if not exists {TABLE_NAME}
                        (document_id int PRIMARY KEY NOT NULL, 
                        title text NOT NULL, 
                        title_h1 text, 
                        tags text, 
                        similar_documents text, 
                        publication_date datetime NOT NULL, 
                        author text NOT NULL,
                        body text NOT NULL, 
                        hyperlink text NOT NULL)''')

    for page_url in tqdm(page_urls):
        sb_document = parse_article_page(page_url)
        sb_document.insert_row_to_sqllite(con, table_name=TABLE_NAME)

    con.close()
