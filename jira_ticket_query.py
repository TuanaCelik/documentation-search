import json
from typing import Optional, List
from haystack import Document, component, Pipeline
from haystack_integrations.components.embedders.jina import JinaTextEmbedder
from haystack_integrations.components.rankers.jina import JinaRanker
from milvus_haystack import MilvusEmbeddingRetriever, MilvusDocumentStore

document_store = MilvusDocumentStore(
    connection_args={
        "host": "localhost",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
)
@component
class RemoveRelated:
    @component.output_types(documents=List[Document])
    def run(self, tickets: List[Document], query_id: Optional[str]):
        retrieved_tickets = []
        for t in tickets:
            if not t.meta['Issue_id'] == query_id and not t.meta['Parent_id'] == query_id:
                retrieved_tickets.append(t)
        return {'documents': retrieved_tickets}
    



retriever = MilvusEmbeddingRetriever(document_store=document_store, top_k=20)
embedder = JinaTextEmbedder(model='jina-embeddings-v2-base-en')
parent_cleaner = RemoveRelated()
ranker = JinaRanker()

query_pipeline_reranker = Pipeline()
query_pipeline_reranker.add_component('query_embedder', embedder)
query_pipeline_reranker.add_component('doc_retriever', retriever)
query_pipeline_reranker.add_component('parent_cleaner', parent_cleaner)
query_pipeline_reranker.add_component('jina_reranker', ranker)

query_pipeline_reranker.connect('query_embedder.embedding', 'doc_retriever.query_embedding')
query_pipeline_reranker.connect('doc_retriever', 'parent_cleaner')
query_pipeline_reranker.connect('parent_cleaner', 'jina_reranker')

with open('tickets.json', 'r') as file:
    tickets = json.load(file)

while(True):
    query_ticket_key = input("Enter ticker number: ")
    for ticket in tickets:
        if ticket['Issue_key'] == query_ticket_key:
            query = str(ticket)
            query_ticket_id = ticket['Issue_id']
            result = query_pipeline_reranker.run(data={'query_embedder':{'text': query},
                                                        'parent_cleaner': {'query_id': query_ticket_id},
                                                        'jina_reranker': {'query': query, 'top_k': 10}
                                                        }
                                                    )

            for idx, res in enumerate(result['jina_reranker']['documents']):
                print('Doc {}:'.format(idx + 1), res)