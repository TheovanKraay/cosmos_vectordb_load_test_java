package com.example;

import com.azure.cosmos.*;
import com.azure.cosmos.implementation.guava25.collect.ImmutableList;
import com.azure.cosmos.models.*;
import com.azure.identity.DefaultAzureCredentialBuilder;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class DataLoaderApp {

    private static final String COSMOS_ENDPOINT = System.getenv("COSMOS_ENDPOINT");
    private static final String DATABASE_NAME = "books";
    private static final String CONTAINER_NAME = "bookitems";
    private static final String DATA_FILE = "staging_embeddings.jsonl";
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) throws Exception {

        CosmosClient client = new CosmosClientBuilder()
                .endpoint(COSMOS_ENDPOINT)
                .credential(new DefaultAzureCredentialBuilder().build())
                .consistencyLevel(ConsistencyLevel.EVENTUAL)
                .buildClient();

        CosmosDatabase database = client.getDatabase(DATABASE_NAME);

        CosmosContainerProperties containerProperties = new CosmosContainerProperties(CONTAINER_NAME, "/partitionKey");

        // Vector embedding policy
        CosmosVectorEmbeddingPolicy vectorEmbeddingPolicy = new CosmosVectorEmbeddingPolicy();
        CosmosVectorEmbedding embedding = new CosmosVectorEmbedding();
        embedding.setPath("/textVector");
        embedding.setDataType(CosmosVectorDataType.FLOAT32);
        embedding.setDimensions(8L);
        embedding.setDistanceFunction(CosmosVectorDistanceFunction.COSINE);
        vectorEmbeddingPolicy.setCosmosVectorEmbeddings(Collections.singletonList(embedding));
        containerProperties.setVectorEmbeddingPolicy(vectorEmbeddingPolicy);

        // Indexing policy
        IndexingPolicy indexingPolicy = new IndexingPolicy();
        indexingPolicy.setIndexingMode(IndexingMode.CONSISTENT);
        indexingPolicy.setExcludedPaths(Collections.singletonList(new ExcludedPath("/*")));
        indexingPolicy.setIncludedPaths(ImmutableList.of(
                new IncludedPath("/fileName/?"),
                new IncludedPath("/text/?")
        ));

        CosmosVectorIndexSpec vectorIndex = new CosmosVectorIndexSpec();
        vectorIndex.setPath("/textVector");
        vectorIndex.setType(CosmosVectorIndexType.DISK_ANN.toString());
        indexingPolicy.setVectorIndexes(Collections.singletonList(vectorIndex));
        containerProperties.setIndexingPolicy(indexingPolicy);

        // Full text indexing
        CosmosFullTextPolicy fullTextPolicy = new CosmosFullTextPolicy();
        CosmosFullTextPath fullTextPath = new CosmosFullTextPath();
        fullTextPath.setPath("/text");
        fullTextPath.setPath("/fileName");
        fullTextPath.setLanguage("en-US");
        fullTextPolicy.setPaths(Collections.singletonList(fullTextPath));
        containerProperties.setFullTextPolicy(fullTextPolicy);

        CosmosFullTextIndex fullTextIndex = new CosmosFullTextIndex();
        fullTextIndex.setPath("/text");
        fullTextIndex.setPath("/fileName");
        indexingPolicy.setCosmosFullTextIndexes(Collections.singletonList(fullTextIndex));

        database.createContainerIfNotExists(containerProperties, ThroughputProperties.createManualThroughput(10000));

        CosmosContainer container = database.getContainer(CONTAINER_NAME);

        // Prepare CosmosItemOperations for bulk
        List<CosmosItemOperation> bulkOperations = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(Paths.get(DATA_FILE))) {
            String line;
            while ((line = br.readLine()) != null) {
                JsonNode node = mapper.readTree(line);
                Map<String, Object> doc = new HashMap<>();
                doc.put("id", UUID.randomUUID().toString());
                doc.put("partitionKey", node.get("partitionKey").asText());
                doc.put("fileName", node.get("fileName").asText());
                doc.put("text", node.get("text").asText());

                JsonNode vectorNode = node.get("textVector");
                List<Float> vector = new ArrayList<>();
                for (JsonNode v : vectorNode) {
                    vector.add((float) v.asDouble());
                }
                doc.put("textVector", vector);

                CosmosItemOperation op = CosmosBulkOperations.getCreateItemOperation(
                        doc,
                        new PartitionKey(doc.get("partitionKey"))
                );

                bulkOperations.add(op);
            }
        }

        System.out.printf("Starting bulk insert of %d documents%n", bulkOperations.size());

        Iterable<CosmosBulkOperationResponse<Object>> responses = container.executeBulkOperations(bulkOperations);

        int successCount = 0;
        int failCount = 0;

        for (CosmosBulkOperationResponse<Object> response : responses) {
            if (response.getException() == null) {
                successCount++;
            } else {
                failCount++;
                response.getException().printStackTrace();
            }
        }

        System.out.printf("Bulk insert complete. Success: %d, Failures: %d%n", successCount, failCount);
        client.close();
    }
}
