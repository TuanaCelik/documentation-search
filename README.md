# Jira Ticket Creation with ReRanking

This is a simple demonstration repo that builds Haystack pipelines that can 
- Index Jira tickets into a `MilvusDocumentStore` 
- Given a "new" Jira ticket, can tell you which other tickets are the most related.
    - This step uses the new Jina ReRanker models

> This repository is based on the [ReRankers with Jina AI and Haystack example by Jina AI](https://jina.ai/news/retrieve-jira-tickets-with-jina-reranker-and-haystack-20/)