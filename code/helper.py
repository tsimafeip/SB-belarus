import os
import re
import hashlib
import datetime
import sqlite3
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

from sb_document import SbDocument
from sb_scraper import collect_links, parse_article_page

# nltk.download("stopwords")
mystem = Mystem()
russian_stopwords = set(stopwords.words("russian"))
russian_stopwords.update({'оно', 'этого', 'это', 'который', 'весь', 'также'})


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


def create_db_table(con, table_name):
    # creates table if it does not exist
    con.cursor().execute(f'''create table if not exists {table_name}
                        (document_id int PRIMARY KEY NOT NULL, 
                        title text NOT NULL, 
                        title_h1 text, 
                        tags text, 
                        similar_documents text, 
                        publication_date datetime NOT NULL, 
                        author text NOT NULL,
                        body text NOT NULL, 
                        hyperlink text NOT NULL)''')


def count_most_common_tags(sb_documents: List[SbDocument]):
    earliest_date = datetime.date.today()
    overall_tags_counter = Counter()
    before_tags_counter = Counter()
    after_tags_counter = Counter()
    revolution_summer_starts = datetime.date(year=2020, month=5, day=15)
    for i, sb_document in enumerate(sb_documents):
        for tag in sb_document.tags:
            overall_tags_counter[tag] += 1
            if sb_document.publication_date < revolution_summer_starts:
                before_tags_counter[tag] += 1
                if sb_document.publication_date < earliest_date:
                    earliest_date = sb_document.publication_date
            else:
                after_tags_counter[tag] += 1

    print(earliest_date)
    print(sum(overall_tags_counter.values()), overall_tags_counter.most_common(10))
    print(sum(before_tags_counter.values()), before_tags_counter.most_common(10))
    print(sum(after_tags_counter.values()), after_tags_counter.most_common(10))


def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(u'\xa0|\n', ' ', text)
    text = re.sub('[^а-яa-z ]', '', text)

    tokens = mystem.lemmatize(text)
    tokens = [token for token in tokens if
              ((token not in russian_stopwords) and (len(token) > 2) and (token.strip() not in punctuation))]

    return tokens


def load_articles_from_links(links_filename: str, db_name: str, table_name: str):
    page_urls = collect_links(path_to_file=links_filename)

    con = sqlite3.connect(db_name)

    for page_url in tqdm(page_urls):
        sb_document = parse_article_page(page_url)
        sb_document.insert_row_to_sqllite(con, table_name=table_name)

    con.close()


def load_data_from_db(db_name: str, table_name: str) -> List[SbDocument]:
    connection = sqlite3.connect(db_name)

    cur = connection.cursor()

    # TODO: Fix hardcode
    col_names = [col_name[0] for col_name in
                 cur.execute("SELECT name FROM pragma_table_info('documents') ORDER BY cid")
                 ]

    docs = []

    for record in cur.execute(f"SELECT * from {table_name};"):
        sb_document_dict = dict(zip(col_names, record))
        sb_document = SbDocument(**sb_document_dict)
        docs.append(sb_document)

    return docs


def preprocess_docs(sb_documents: List[SbDocument], target_filename: str = '../data/preprocessed_docs.txt') -> List[
    str]:
    words = []
    with open(target_filename, 'w') as f:
        f.write(str(len(sb_documents)) + '\n')
        for sb_document in tqdm(sb_documents):
            preprocessed_body = []
            for line in sb_document.body.splitlines(keepends=False):
                if line:
                    preprocessed_tokens = preprocess_text(line)
                    preprocessed_text = " ".join(preprocessed_tokens)
                    preprocessed_body.append(preprocessed_text)
                    words.extend(preprocessed_tokens)
            f.write(" ".join(preprocessed_body) + '\n')

    return words


def read_preprocessed_docs(source_filename: str = '../data/preprocessed_docs.txt') -> Tuple[List[str], List[str]]:
    words = []
    docs = []
    with open(source_filename, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if i == 0:
                continue

            docs.append(line.strip())
            words.extend(docs[-1].split())

    return docs, words


def count_articles_by_month(sb_documents: List[SbDocument], n: int = 5):
    month_buckets = Counter()
    cur_date = (sb_documents[0].publication_date.month, sb_documents[0].publication_date.year)
    buckets = [0]
    dates = [cur_date]
    for i, sb_document in enumerate(sb_documents):
        month_year_pair = (sb_document.publication_date.month, sb_document.publication_date.year)
        if month_year_pair != cur_date:
            cur_date = month_year_pair
            buckets.append(1)
            dates.append(month_year_pair)
        else:
            buckets[-1] += 1
        month_buckets[month_year_pair] += 1

    draw_with_ticks(x=np.arange(len(buckets)), y=np.array(buckets), labels=dates)

    print(month_buckets.most_common(n))
    print(sum(month_buckets.values()) / len(month_buckets))


def draw_with_ticks(x, y, labels):
    # set figure size and dpi (dots per inch)
    plt.figure(figsize=(10, 6), dpi=80)

    # Create a new subplot from a grid of 1x1
    plt.subplot(111)

    # customize color and line width
    plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-")

    # set lower & upper bound by taking min & max value respectively
    plt.xlim(x.min() * 1.1, x.max() * 1.1)
    plt.ylim(y.min() * 1.1, y.max() * 1.1)

    # provide five tick values for x and 3 for y
    # and pass the corresponding label as a second argument.
    x_ticks = []
    x_tick_labels = []
    for x_val, x_label in zip(x, labels):
        if x_label[0] == 1:
            x_ticks.append(x_val)
            x_tick_labels.append(x_label[1])

    plt.xticks(x_ticks, x_tick_labels)

    plt.yticks([100, 200, 300, 400, 500])

    plt.grid(True)

    path_to_image = '../meta/article_num.png'

    if not os.path.isfile(path_to_image):
        plt.savefig(path_to_image, format='png', bbox_inches='tight')

    # show it on screen
    plt.show()


def run_preliminary_analysis(sb_documents: List[SbDocument]):
    # most popular tags
    count_most_common_tags(sb_documents)

    # articles per month
    count_articles_by_month(sb_documents, 10)

    # words = preprocess_docs(sb_documents)
    _, full_words = read_preprocessed_docs()
    _, after_words = read_preprocessed_docs('../data/after_docs.txt')
    _, before_words = read_preprocessed_docs('../data/before_docs.txt')

    # most popular words
    full_words_counter = Counter(full_words)
    before_words_counter = Counter(before_words)
    after_words_counter = Counter(after_words)
    print(len(full_words), [(token, count / len(full_words)) for token, count in full_words_counter.most_common(10)])
    print(len(before_words),
          [(token, count / len(before_words)) for token, count in before_words_counter.most_common(10)])
    print(len(after_words), [(token, count / len(after_words)) for token, count in after_words_counter.most_common(10)])

    # num of unique tokens
    print('Num of tokens: ', len(full_words))
    print('Num of unique tokens: ', len(full_words_counter))


def parse_parquet_to_sqllite(db_name: str, table_name: str, source_path: str = "../../NewsAnalysis-master/data"):
    filepaths = []

    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(".parquet"):
                filepaths.append(os.path.join(root, file))

    frames = []
    for filepath in filepaths:
        frames.append(pd.read_parquet(filepath, engine='fastparquet'))

    cnx = sqlite3.connect(db_name)
    result = pd.concat(frames)
    result.applymap(str).to_sql(name=table_name, con=cnx)
