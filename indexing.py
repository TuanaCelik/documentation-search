import json
from typing import List
from haystack import Document, component, Pipeline
from milvus_haystack import MilvusDocumentStore
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder

document_store = MilvusDocumentStore(
    connection_args={
        "host": "localhost",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
)

relevant_keys = ['Summary', 'Issue_key', 'Issue_id', 'Parent_id', 'Issue type', 'Status', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Last Viewed', 'Due Date', 'Labels',
                 'Description', 'Comment', 'Comment__1', 'Comment__2', 'Comment__3', 'Comment__4', 'Comment__5', 'Comment__6', 'Comment__7', 'Comment__8', 'Comment__9', 'Comment__10', 'Comment__11', 'Comment__12',
                 'Comment__13', 'Comment__14', 'Comment__15']

@component
class RemoveKeys:
    @component.output_types(documents=List[Document])
    def run(self, file_name: str):
        with open(file_name, 'r') as file:
            tickets = json.load(file)
        cleaned_tickets = []
        for t in tickets:
            t = {k: v for k, v in t.items() if k in relevant_keys and v}
            cleaned_tickets.append(t)
        return {'documents': cleaned_tickets}
    
@component
class JsonConverter:
    @component.output_types(documents=List[Document])
    def run(self, tickets: List[Document]):
        tickets_documents = []
        for t in tickets:
            if 'Parent id' in t:
                t = Document(content=json.dumps(t), meta={'Issue_key': t['Issue_key'], 'Issue_id': t['Issue_id'], 'Parent_id': t['Parent_id']})
            else:
                t = Document(content=json.dumps(t), meta={'Issue_key': t['Issue_key'], 'Issue_id': t['Issue_id'], 'Parent_id': ''})
            tickets_documents.append(t)
        return {'documents': tickets_documents}

cleaner = RemoveKeys()
converter = JsonConverter()
embedder =  JinaDocumentEmbedder(model='jina-embeddings-v2-base-en')
writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component('cleaner', cleaner)
indexing_pipeline.add_component('converter', converter)
indexing_pipeline.add_component('embedder', embedder)
indexing_pipeline.add_component('writer', writer)

indexing_pipeline.connect('cleaner', 'converter')
indexing_pipeline.connect('converter', 'embedder')
indexing_pipeline.connect('embedder', 'writer')

indexing_pipeline.run({'cleaner': {'file_name': 'tickets.json'}})