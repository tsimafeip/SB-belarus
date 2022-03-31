import os
import json

from typing import Optional, List, Union
from datetime import datetime


class SbDocument:
    def __init__(self, document_id: int, title: str, publication_date: Union[datetime, str],
                 author: str, body: str, hyperlink: str,
                 tags: Optional[Union[str, List[str]]] = None, title_h1: Optional[str] = None,
                 similar_documents: Optional[Union[str, List[str]]] = None):
        """Initialises document object."""
        # deserialize from db values
        if isinstance(tags, str):
            tags = self._deserealise_list_from_str(tags)
        if isinstance(similar_documents, str):
            similar_documents = self._deserealise_list_from_str(similar_documents)
        if isinstance(publication_date, str):
            publication_date = datetime.strptime(publication_date, "%Y-%m-%dT%H:%M:%S%z").date()

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

    @classmethod
    def _serealise_list_to_str(cls, field: Optional[List[str]]) -> Optional[str]:
        return "@".join(field) if field else None

    @classmethod
    def _deserealise_list_from_str(cls, field: Optional[str]) -> Optional[List[str]]:
        return field.split('@') if field else None

    def insert_row_to_sqllite(self, connection, table_name: str):
        params_dict = vars(self)
        params_dict['similar_documents'] = self._serealise_list_to_str(self.similar_documents)
        params_dict['tags'] = self._serealise_list_to_str(self.tags)
        connection.cursor().execute(f'''insert into {table_name} values (:document_id, :title, :title_h1, :tags,
                    :similar_documents, :publication_date, :author, :body, :hyperlink)''', params_dict)
        connection.commit()

    @staticmethod
    def load_from_json(document_id: int) -> 'SbDocument':
        with open(os.path.join(f'{document_id}.json'), 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            return SbDocument(**data_dict)
