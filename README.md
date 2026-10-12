# RAG
Since you're building your AI/ML roadmap from NumPy → Pandas → SQL → Scikit-learn → Deep Learning → LLMs → PyTorch, RAG (Retrieval-Augmented Generation) is an excellent next topic to study.

For your goal of becoming an AI/ML Engineer, don't learn RAG as just a LangChain tutorial. Learn the underlying retrieval algorithms, data pipeline, evaluation methods, system architecture, and production engineering.

# RAG roadmap for Data Science, AI & ML

[Building a RAG Pipeline from Scratch: A Visual Guide to Ingestion & Retrieval | by Abhishek Galfade | Medium](https://images.openai.com/static-rsc-4/eek7hFofXTvmXfFFSGRZhjSt9JIY-TI8-J-eW7WUD65LckZt77jqOiwv_GrIAdxfO224zQtqVNHPAmAoCwSvrCrsz-VqQd6GBTXfaefa7BeX0SFyl-WscXbwSNAESp5w-wEW7Frif9N_Ggz7JWI9WWOJdZJ2J08-9QZb4vQVFLM?purpose=inline)

1\. Data ingestion and preparation

Load, clean, parse, normalize and prepare documents.

[Indexação de Documentos por IA Explicada](https://images.openai.com/static-rsc-4/quF5SlF89mti6UR8ns-qlcZYLFfW0-04NCCaeBGKkVSNxeQqVHYeggp8jVDa58ho0SzkvPXPk8M35WeM1Z1EP6amNbb1V-ouj3UzvBEaTVYshaBctSeJ2HwRUGy7EK8yLDXszgUv3cTWhUoBQln9s0v-BDuw6Hp2289zxBGIpA4?purpose=inline)

2\. Chunking and embeddings

Split documents and represent their meaning as vectors.

[Building a Chatbot with RAG: How Retrieval Meets the LLM | Yong Sen](https://images.openai.com/static-rsc-4/AUZunEa0t3gOuzwgtcj0KnGXUeyQk0OghM5JgpAmFhyXesbKuJLy7BkvP-AB74UVPPpfJlEPGsMhW8BFH5cexSgURah4ZKrBgm-wWPdljZBfU2yCfhEHNTqf4z6qGrPiR5WwdRsxamoeTdhzQusvRFdEFWmPYS7kfJDuIJUZkl8?purpose=inline)

3\. Retrieval and reranking

Find and rank the most relevant information for a query.

[The Complete Guide to RAG (Retrieval Augmented Generation) | by Ritesh Biradar | Medium](https://images.openai.com/static-rsc-4/Vi0XfmmwbhTpq12Be1u7XpQHClySAO_6ccYmXjF8Ed-OUqg-3O4M20pmtFrF4-wuE5BW0jUbWNgASgPLFz9zrGtr1L9ib3I6WRP82wyN36znbeE5Axlw3HpkDkJMLEu9K0DOXLsUkA0ZF9PlrYuHk60_XIugPT9QqFH-09LfYFQ?purpose=inline)

4\. Generation and grounding

Give retrieved context to an LLM to generate evidence-based answers.

[The Hidden Art of RAG Evaluation: Why 90% of AI Teams Get It Wrong (And How to Be in the Top 10%) | by Abhishek Pan | Medium](https://images.openai.com/static-rsc-4/Kf5OJ7SwkuzhpFT6T5xs65O6gFLlwJpUUS7iFOAVAbguYQrRMvf9sZrfMUiBpDEjvzwqK7QG_LZ2AyJ2yZ7WUgWM_IxbKGjZyWS_LEgTCfJdbNZKOeyMWA3ZA4L2wSBbV-cHTszf7bRuTiyo1Pn3a4QY5TDZllLEZhfo-1uV3q4?purpose=inline)

5\. Evaluation and production

Measure quality, reduce errors, control costs and deploy reliably.

## 1. RAG fundamentals ⭐⭐⭐⭐⭐

Understand these concepts before writing code.

- What is Retrieval-Augmented Generation?
- Why use RAG instead of an LLM alone?
- RAG vs fine-tuning vs prompt engineering
- Knowledge-intensive tasks
- Parametric vs external knowledge
- Retrieval, augmentation and generation
- Grounding and citations
- Hallucinations and knowledge limitations
- Offline indexing vs online query processing
- RAG architecture and its limitations

Key distinction: RAG normally retrieves relevant information at query time; it doesn't inherently retrain or modify the LLM's weights.

## 2. Data ingestion and document processing ⭐⭐⭐⭐⭐

RAG quality depends heavily on the quality of the source data.

Learn:

- Document loaders
- PDF, DOCX, TXT, HTML, Markdown and CSV parsing
- Web pages and knowledge bases
- OCR for scanned documents
- Table extraction
- Metadata extraction
- Text normalization
- Removing duplicate documents
- Handling headers, footers and boilerplate
- Document IDs and versioning
- Incremental ingestion and updates
- Access permissions and document filtering

Understand the difference between raw documents, parsed documents, chunks and metadata.

For data science applications, learn how to ingest structured data from databases as well as unstructured text.

## 3. Chunking strategies ⭐⭐⭐⭐⭐

Chunking determines how documents are divided into retrievable units.

[📘 Chunking Strategies for RAG and Vector Databases](https://images.openai.com/static-rsc-4/uyqzfBFx0f53qY0LSwQw00uAIaeVUicnBJBsEw9dVqt4bnFt9Tb6lFKSmZTp_i1xswch1VYM26iJqyEiypUCuaLiSgomdzxm_hDAogu52x3TYT82_i19UzYIh6G_vQC7QHxm3HdSmxKk6urD5gGNhbHFMIpA9yFLbwfE-0PCQ70?purpose=inline)

Fixed-size chunking

Split text by a specified number of tokens or characters. Simple baseline, but may split ideas in the middle.

[The Art and Science of Text Chunking for Better AI Retrieval | by Priyanka Dave | Medium](https://images.openai.com/static-rsc-4/3ArIZd8TIhV5AP8j9Qw4TsOJ7BTS8QaZgODCAGk3J88NExe5zCkYOG8D8I8qLkY9XqnbCptOyIgnLUor-tEcelU2N90Sw_sAn11OWHO5TVXTjsNHvSUdoM1GvzQTl1OkU3SsA-oHWpDeMBGt60Gm2QCmLtuoYqBGVs8HXQ-aBao?purpose=inline)

Recursive chunking

Split using a hierarchy of separators, such as sections, paragraphs and sentences.

[Semantic Chunking. Chunking is where retrieval quality is… | by Pranshu Vashishth | Mar, 2026 | Medium](https://images.openai.com/static-rsc-4/mei73zFzQA2rKdz2-OlziApp8hinbBte-0wZjLgyfcRR84niDZdzw12mHQWBm0SUGvKYLAjktRoqt4_azmYswHAJBfLFPO7DklC8TwmA2JtWNhGLJmcFgWbEk-x3-sMFulZ0OqWNjNUfVvmQkGoEvqct4TyOX2gdY271t8iabkA?purpose=inline)

Semantic chunking

Use changes in meaning to help determine chunk boundaries.

Also study:

- Sentence-based chunking
- Structure-aware chunking
- Chunk size and overlap
- Token limits
- Parent-child chunking
- Contextual chunking
- Table-aware chunking

Practical experiment: Compare several chunk sizes and overlap settings on the same dataset. Measure retrieval quality rather than assuming that larger or smaller chunks are always better.

## 4. Embeddings ⭐⭐⭐⭐⭐

Embeddings convert text into numerical vectors that capture aspects of its meaning.

Learn:

- What embeddings represent
- Embedding models
- Dense vectors
- Embedding dimensions
- Cosine similarity
- Dot-product similarity
- Euclidean distance
- Query and document embeddings
- Similarity vs relevance
- Domain-specific embeddings
- Multilingual embeddings
- Embedding model selection
- Embedding normalization
- Embedding model versioning

Example:

```
Document chunk
      ↓
Embedding model
      ↓
[0.12, -0.34, 0.81, ...]
      ↓
Stored in vector index
```

Understand that two semantically similar texts may have nearby vectors, but vector similarity does not guarantee that a passage answers a question correctly.

## 5. Vector databases and vector search ⭐⭐⭐⭐⭐

Learn the underlying concepts before choosing a product.

| Concept                             | Why it matters                                             |
| ----------------------------------- | ---------------------------------------------------------- |
| Vector index                        | Organizes vectors for efficient search                     |
| Exact nearest-neighbor search       | Finds the true nearest vectors, often expensive at scale   |
| Approximate nearest neighbors (ANN) | Trades some search accuracy for speed                      |
| HNSW                                | A popular graph-based ANN indexing method                  |
| IVF                                 | Partitions vectors into clusters for search                |
| Metadata filtering                  | Restricts results by source, date, permissions or category |
| Upsert and delete                   | Keeps the index synchronized with source data              |
| Hybrid search                       | Combines lexical and semantic retrieval                    |

Choose one technology to implement first:

- FAISS — useful for learning vector search locally.
- Qdrant — useful for learning vector database workflows.
- pgvector — useful when you want PostgreSQL plus vector search.
- Milvus or Weaviate — alternatives to explore as your needs grow.

You don't need to master every database at the beginning.

## 6. Retrieval algorithms ⭐⭐⭐⭐⭐

This is one of the most important areas for an ML-focused RAG engineer.

### A. Sparse retrieval

Learn:

- TF-IDF
- BM25
- Inverted indexes
- Keyword matching

Best for exact names, identifiers and technical terms.

### B. Dense retrieval

Learn:

- Embedding-based retrieval
- Cosine similarity
- Top-k retrieval
- Query-document similarity

Useful when a question uses different wording from the source.

### C. Hybrid retrieval

Combine sparse and dense results.

Learn:

- Score normalization
- Reciprocal Rank Fusion (RRF)
- Candidate merging
- Ranking trade-offs

### D. Advanced retrieval

- Metadata filtering
- Query rewriting
- Multi-query retrieval
- Hypothetical Document Embeddings (HyDE)
- Query decomposition
- Context expansion
- Parent-document retrieval

A crucial principle: better retrieval can improve RAG more than simply switching to a larger LLM. Test this on your actual dataset.

## 7. Reranking ⭐⭐⭐⭐⭐

Initial retrieval finds candidate chunks. A reranker then scores those candidates more carefully.

Learn:

- Bi-encoder vs cross-encoder
- Candidate retrieval
- Reranking scores
- Top-k selection
- Latency vs relevance trade-offs
- Reranking before context construction

Typical pipeline:

```
User query
    ↓
Retrieve 20–50 candidates
    ↓
Reranker scores query + candidate pairs
    ↓
Select the best few chunks
    ↓
Send context to LLM
```

The counts above are illustrative, not fixed rules.

Study when reranking improves quality enough to justify its additional compute and latency.

## 8. Prompt construction and generation ⭐⭐⭐⭐⭐

Once you retrieve context, you must construct a reliable prompt.

Learn:

- System instructions
- User queries
- Context placement
- Context ordering
- Citation requirements
- Structured output
- Context-window limits
- Handling missing evidence
- Refusal or abstention when context is insufficient
- Conflicting sources
- Prompt injection through retrieved documents

A basic grounded prompt should tell the LLM to answer from the provided evidence, cite the supporting sources, and acknowledge when the evidence is insufficient.

Remember: instructions alone cannot guarantee that a model will never hallucinate. You must evaluate and validate its answers.

## 9. RAG evaluation ⭐⭐⭐⭐⭐

This is essential if you want to move beyond tutorial-level projects.

Evaluate retrieval and generation separately.

### Retrieval evaluation

- Recall\@k: Did the retrieved results include the relevant evidence?
- Precision\@k: How many retrieved results were relevant?
- MRR: How high was the first relevant result ranked?
- nDCG\@k: How well were results ranked, accounting for graded relevance?

### Generation evaluation

- Faithfulness: Is the answer supported by the retrieved context?
- Answer relevance: Does it answer the actual question?
- Correctness: Does it match a trusted reference answer?
- Citation quality: Do cited sources support the claims?

### System evaluation

- Latency
- Cost per query
- Tokens consumed
- Failure rate
- User satisfaction

Also learn:

- Creating a representative test set
- Human evaluation
- LLM-as-a-judge and its limitations
- Ablation testing
- Regression testing
- Retrieval and generation error analysis

Tools worth exploring include Ragas, DeepEval and TruLens. Learn the metrics first so you understand what the tools are measuring.

## 10. Advanced RAG ⭐⭐⭐⭐⭐

Once you can build and evaluate basic RAG, move on to:

- Query expansion and rewriting
- Hybrid retrieval
- Reranking
- Context compression
- Parent-child retrieval
- Multi-hop retrieval
- Multi-query retrieval
- Metadata-aware retrieval
- Adaptive retrieval
- Agentic RAG
- Graph RAG
- Multimodal RAG

Don't assume advanced methods are automatically better. Establish a baseline, change one component at a time, and measure the improvement.

## 11. RAG with structured and tabular data ⭐⭐⭐⭐

This is particularly relevant to data science and enterprise AI.

Learn how to answer questions using:

- CSV and Excel files
- SQL databases
- Tables in PDFs
- Product catalogs
- Knowledge graphs
- Business metrics and dashboards

Understand the difference between:

Text retrieval — finding passages in documents.

SQL-based answering — translating questions into database queries.

Hybrid systems — using retrieval for explanations and SQL for precise calculations.

For example, “What is our revenue this quarter?” may require a validated SQL query, not semantic search over text chunks.

## 12. Frameworks and tools ⭐⭐⭐⭐⭐

Learn the fundamentals first, then use a framework to speed up development.

[LangChain - Partner Ecosystem | MongoDB](https://images.openai.com/static-rsc-4/XFgDhJXhGfSAGUFZeG8KDYHPCvHBDtPNtRoNH4-j_ThOcv1TfVbAv-vLM99Wi5g6yksscqygP9bXC-9EjIvfq92LVcR3rRSCWZUT5FxM-XUoAzdtQcikAXJmB0LMbidgqKoKHs5xWVsQnIHyEF43mufdsEU5FehmA_h-FIVEWQo?purpose=inline)

LangChain

Document loaders, splitters, retrievers, prompt templates, chains and integrations.

[LlamaIndex IA - Construye agentes LLM sobre datos empresariales](https://images.openai.com/static-rsc-4/Jwyz2pGQx3aEvo88hfCxFHaSOdnH8jabqMSMBe9v1eQIxy_-qmfvCmMNFdzPKNgHOOxI3pMSShPq0BfaSJLeoet3PftCEu4tC6zcFl-iHXXVAcWa52sLB-JnJLWzuvBg20TFbjOmD9XuWqVoTWMy6WpMrqizWJHRCOACUFk6OQw?purpose=inline)

LlamaIndex

Data ingestion, indexing, retrieval and query engines.

[How to Use Hugging Face Transformers with PyTorch in Python | by The Analytics Edge 🌟 | Medium](https://images.openai.com/static-rsc-4/yqyUO_VWMttCxj60lAAhKaltjSTcPj6bTSen3gsaridryI_kYxcxqIC-fyZAb81nPjnMAGz9Q68wVen-cqk2v6mZXRML114QwOpzJcb_0lbjD7VcyqGk8ZqCZ5AufGDr5kC_pbIWXBLhvB0b_UNru6AntkM9bpvZ2LXJxOVbjPE?purpose=inline)

Hugging Face Transformers

Model and tokenizer access, embedding models, rerankers and generation.

[Qdrant Logo PNG Vector (SVG) Free Download](https://images.openai.com/static-rsc-4/Ais5_PO1Q56lnD04G32Sa9OvzjljEw3O6tdYc-8PMK7p_5K9QpkMAg5Pmk0uI4UzILesmZm2eR-gVp2kGg1yvxjYFlu6fPcQk4i5V2hvE2OTpew59H-VjVlQKDxKl4VKyLCWREm2VDQxsKr2Pb4spxLXKRRt6ijY6juSYnsWlu4?purpose=inline)

Qdrant

Vector storage, similarity search, payload metadata and filtering.

Also become comfortable with Python, API calls, environment variables, FastAPI, and basic testing.

Recommendation: Build one small RAG pipeline directly with an embedding model, vector search and an LLM API before adopting a framework. This makes debugging much easier.

## 13. Production-ready RAG ⭐⭐⭐⭐⭐

A working demo is different from a reliable application.

Learn:

- API design and request validation
- Authentication and authorization
- Document-level access control
- Incremental indexing
- Caching
- Async processing
- Retries and timeouts
- Rate limiting
- Logging and tracing
- Monitoring
- Versioning prompts, models and indexes
- Cost and latency optimization
- Data privacy and retention
- Backup and recovery
- Evaluation in CI/CD

Security deserves special attention: retrieved documents are data, not trusted instructions. Prevent unauthorized document retrieval and test for prompt injection.

# 14. Projects to build in order

(https://images.openai.com/static-rsc-4/trEfY8KmBe-KdIeRAgJIm9nWMgO-LnM2wwBJyQICNYhAP-B7QRH---DSUl6zTaqdIDDYscnLrCdQItONBF9V0GAFF8RLPn2nGJ4hrROJtO5B3iU3oRzYKTNl4OIMVI2NRrz1iz-usVPZ3nPpW4IFzQs3LILZphSua6QXQ5U8a9E?purpose=inline)

Project 1 — PDF Q&A assistant

Beginner

Build document loading, chunking, embeddings, vector search and answers with citations.

Demonstrates: basic RAG pipeline.

[GitHub - introfini/ZotSeek: AI semantic search for Zotero, with a built-in MCP server for AI agents (Claude Code, Codex). Find papers by meaning. 100% local and private. · GitHub](https://images.openai.com/static-rsc-4/q2BEA1AjdcL867F0NwrKzRfEkcgCtL9ItRZlJmpEgi2hotusrIwMHyGR7_Bc_y3_y9gdTi6SetNM4L7fMMMahnRNbViLerBzimOaRc1uLHJXUx8GtwJVeVXfnX626-dvQw6qWPYQq2RgsBdQ7HjmiIVYGEo6tOQboGuTVcZLiZk?purpose=inline)

Project 2 — Research-paper assistant

Intermediate

Search papers using hybrid retrieval, reranking and source references.

Demonstrates: retrieval algorithms and evaluation.

[Cloudera Data Viz AI Assistant Demo - Cloudera Videos](https://images.openai.com/static-rsc-4/lqT9LuYkf0UvV6TAvi178ifK8SeL_ptiRLJU0Rb2mpC_Y880d0XZTnKaxMNK-uZ7qBeAP48pi-eIBejM8H-ClNnmX3MsjVMLFsCvqaT73G8f7lw5HokGNWeVKfbIRP21D20sOquhu6FgLXL_9O9R0mFintcdoas4QvCl4WCt5zs?purpose=inline)

Project 3 — Data analyst assistant

Intermediate

Answer questions using SQL for calculations and RAG for documentation and definitions.

Demonstrates: structured data, query validation and grounding.

[Building Our Own LLM Assistant — Development Seed](https://images.openai.com/static-rsc-4/E4x9FUgDDvuq4Z5I_xNVxifp15pkpwjY0V-W-wg9P9pEGP4FLuRxo5gV_oalaTjuI_pwmhMFytxTbzbei28UAQlhA3S_y0pxkzo0NHvEHNh4hqPrhmmfB_JyOypwFD2FdsDGK_CJscnOq8KX3apkxvTbqS05B9DpS83MZowGe9A?purpose=inline)

Project 4 — Production-style RAG API

Advanced

Add FastAPI, access control, evaluation, monitoring, caching and Docker.

Demonstrates: AI engineering and deployment.

## 15. A practical 6-week study plan

| Week | Focus                                            | Deliverable                       |
| ---- | ------------------------------------------------ | --------------------------------- |
| 1    | RAG fundamentals, document ingestion and parsing | Document-processing pipeline      |
| 2    | Chunking, embeddings and vector search           | Semantic search prototype         |
| 3    | Retrieval, hybrid search and reranking           | Improved retrieval system         |
| 4    | LLM prompting, citations and basic RAG           | PDF Q&A application               |
| 5    | Evaluation, failure analysis and advanced RAG    | Evaluation report and comparisons |
| 6    | FastAPI, security, logging and Docker            | Deployable RAG API                |

Aim to spend time on both theory and implementation. If you already understand embeddings and Transformers, you can move faster through the first two weeks.

# Final checklist: What you should master

## RAG learning tracker

0 / 20 completed

Foundations

RAG architecture and limitations

Document ingestion and parsing

Chunking strategies

Embedding models and similarity metrics

Vector databases and ANN search

Retrieval

BM25 and dense retrieval

Hybrid retrieval and RRF

Metadata filtering

Reranking

Query rewriting and multi-query retrieval

Generation and evaluation

Prompt construction and grounding

Citations and abstention

Recall\@k, Precision\@k, MRR and nDCG

Faithfulness and answer relevance

Test datasets and error analysis

Advanced and production

Advanced RAG patterns

Structured data and SQL integration

Frameworks and APIs

Security and access control

Monitoring, latency, cost and deployment

Copy checklist

My recommendation for your AI/ML career: prioritize chunking → embeddings → retrieval → reranking → evaluation → advanced RAG → production deployment. That combination teaches you not only to build a RAG demo, but also to diagnose why it fails and improve it systematically.