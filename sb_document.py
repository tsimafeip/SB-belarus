import os
import json

from typing import Optional, List, Tuple


class SbDocument:
    def __init__(self, document_id: int, title: str, publication_date: str, author: str, body: str, hyperlink: str,
                 tags: Optional[List[str]] = None, title_h1: Optional[str] = None,
                 similar_documents: Optional[List[str]] = None):
        self.document_id = document_id
        self.title = title
        self.title_h1 = title_h1
        self.tags = tags if tags else []
        self.similar_documents = similar_documents if similar_documents else []
        self.publication_date = publication_date
        self.author = author
        self.body = body
        self.hyperlink = hyperlink

    def dump_to_json(self):
        with open(os.path.join(f'{self.document_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(self), f)

    def insert_row_to_sqllite(self, connection, table_name: str):
        params_dict = vars(self)
        params_dict['similar_documents'] = "@".join(self.similar_documents) if self.similar_documents else None
        params_dict['tags'] = "@".join(self.tags) if self.tags else None
        connection.cursor().execute(f'''insert into {table_name} values (:document_id, :title, :title_h1, :tags,
                    :similar_documents, :publication_date, :author, :body, :hyperlink)''', params_dict)
        connection.commit()

    @staticmethod
    def load_from_json(document_id: int) -> 'SbDocument':
        with open(os.path.join(f'{document_id}.json'), 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            return SbDocument(**data_dict)
