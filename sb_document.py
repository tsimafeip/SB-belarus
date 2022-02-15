import os
import json

from typing import Optional, List, Tuple


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
        with open(os.path.join('data', f'{self.document_id}.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(self), f)

    @staticmethod
    def load_from_json(document_id: int) -> 'SbDocument':
        with open(os.path.join('data', f'{document_id}.json'), 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            return SbDocument(**data_dict)
