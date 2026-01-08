# 간단한 BM25/keyword 기반 검색기 (Whoosh 사용)
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from langchain_core.documents import Document
from typing import List
import os

class BM25KeywordRetriever:
    def __init__(self, index_dir: str, content_field: str = "text"):
        self.index_dir = index_dir
        self.content_field = content_field
        self.ix = open_dir(index_dir)
        self.parser = QueryParser(content_field, schema=self.ix.schema)

    def search(self, query: str, k: int = 5) -> List[Document]:
        with self.ix.searcher() as searcher:
            q = self.parser.parse(query)
            results = searcher.search(q, limit=k)
            docs = []
            for hit in results:
                docs.append(Document(
                    page_content=hit[self.content_field],
                    metadata=dict(hit)
                ))
            return docs

# 사용 예시:
# retriever = BM25KeywordRetriever(index_dir="whoosh_index")
# docs = retriever.search("근로기준법 제60조", k=3)
