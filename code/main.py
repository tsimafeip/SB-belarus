import os
import sqlite3

from helper import load_data_from_db, run_preliminary_analysis
import pandas as pd

DB_NAME = '../data/sb_articles.db'
TABLE_NAME = 'documents'

if __name__ == '__main__':
    # sb_documents = sorted(load_data_from_db(DB_NAME, TABLE_NAME), key=lambda x: x.publication_date)
    # run_preliminary_analysis(sb_documents)






    t = 1
