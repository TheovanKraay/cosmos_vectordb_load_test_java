# Cosmos VectorDB Load Test (Java)

This project demonstrates how to load and evaluate vector search, text search, and hybrid search in Azure Cosmos DB for NoSQL using the Java SDK.

---

## Prerequisites

- Java 17+
- Maven
- Azure Subscription
- Cosmos DB for NoSQL account with vector indexing enabled
- Azure OpenAI account for generating embeddings

---

## Configuration

The following environment variables must be set before running any of the apps:

| Variable | Purpose |
| -------- | ------- |
| `COSMOS_ENDPOINT` | Your Cosmos DB endpoint |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key |
| `OPENAI_ENDPOINT` | Azure OpenAI endpoint |

---

## Modules Overview

### 1️⃣ DataLoaderApp.java

- Loads data from `staging_embeddings.jsonl` into Cosmos DB using vector, text, and full-text indexing.
- Also applies correct indexing policies.
- Creates the container if it doesn't exist.

**Run:**

```bash
mvn compile exec:java -Dexec.mainClass="com.example.DataLoaderApp"
```

---

### 2️⃣ VectorSearchApp.java

- Provides methods to run:
  - Vector Search
  - Text Search (full-text)
  - Hybrid Search using Reciprocal Rank Fusion (RRF)

This is called by the evaluation app for search functionality.

---

### 3️⃣ EvaluationApp.java

- Evaluates search performance against predefined queries and ground truth.
- Computes recall for Vector, Text, and Hybrid search.
- Loads queries from `books_processing/questions/` directory.

**Run:**

```bash
mvn compile exec:java -Dexec.mainClass="com.example.EvaluationApp"
```

> Make sure your current directory has the correct relative path to `books_processing/questions/`.

---

### 4️⃣ BulkDataLoaderApp.java

- Similar to `DataLoaderApp` but uses Cosmos DB SDK Bulk API for much faster ingestion.
- Useful for loading large datasets.

**Run:**

```bash
mvn compile exec:java -Dexec.mainClass="com.example.BulkDataLoaderApp"
```

---

## Dataset Notes

- `staging_embeddings.jsonl` contains documents with text and precomputed OpenAI embeddings.
- `books_processing/questions/` contains evaluation questions and corresponding ground-truth files.

---

## Notes

- You may need to adjust Cosmos DB throughput (RU/s) for faster loading and searching.
- The bulk loader uses partition key `"books_items"`, ensure your input data matches this.

---