package com.example;

import com.azure.ai.openai.OpenAIAsyncClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.ai.openai.models.EmbeddingItem;
import com.azure.ai.openai.models.EmbeddingsOptions;
import com.azure.core.credential.AzureKeyCredential;
import com.azure.core.http.policy.ExponentialBackoffOptions;
import com.azure.core.http.policy.RetryOptions;
import com.azure.cosmos.*;
import com.azure.cosmos.models.*;
import com.azure.cosmos.util.CosmosPagedIterable;
import com.azure.identity.DefaultAzureCredentialBuilder;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.*;


public class VectorSearchApp {
    private static final String COSMOS_ENDPOINT = System.getenv("COSMOS_ENDPOINT");
    private static final String DATABASE_NAME = "books";
    private static final String CONTAINER_NAME = "bookitems";
    private static final String OPENAI_API_KEY = System.getenv("AZURE_OPENAI_API_KEY");
    private static final String OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002";
    private static final String OPENAI_EMBEDDING_URL = System.getenv("OPENAI_ENDPOINT");

    private static final ObjectMapper mapper = new ObjectMapper();

    public static OpenAIAsyncClient openAIClient;

    public static void main(String[] args) throws Exception {

        String searchQuery = "Give me a list of books published in the year 2000"; // Default search query if none provided
        System.out.println("Search Query: " + searchQuery);
        RetryOptions retryOptions = new RetryOptions(
                new ExponentialBackoffOptions()
                        .setMaxRetries(10)
                        .setMaxDelay(Duration.of(2, ChronoUnit.SECONDS)));

        openAIClient = new OpenAIClientBuilder()
                .endpoint(OPENAI_EMBEDDING_URL)
                .credential(new AzureKeyCredential(OPENAI_API_KEY))
                .retryOptions(retryOptions)
                .buildAsyncClient();

        List<Float> embedding = getEmbedding(searchQuery);

        CosmosClient client = new CosmosClientBuilder()
                .endpoint(COSMOS_ENDPOINT)
                .credential(new DefaultAzureCredentialBuilder().build())
                .buildClient();

        CosmosContainer container = client.getDatabase(DATABASE_NAME).getContainer(CONTAINER_NAME);

        vectorSearch(container, embedding, 5, 0.7);
        textSearch(container, searchQuery, 5);
        hybridSearchRRF(container, embedding, searchQuery, 5);

        client.close();
    }

    public static List<Float> getEmbedding(String query) {
        try {
            EmbeddingsOptions options = new EmbeddingsOptions(List.of(query));
            options.setUser("");

            var response = openAIClient.getEmbeddings(OPENAI_EMBEDDING_MODEL, options).block();

            List<EmbeddingItem> embeddings = response.getData();

            return embeddings.get(0).getEmbedding().stream().toList();
        } catch (Exception ex) {
            System.err.println("Error getting embedding: " + ex.getMessage());
            ex.printStackTrace();
            return null;
        }
    }


    public static List<String> vectorSearch(CosmosContainer container, List<Float> embedding, int topK, double threshold) {
        System.out.println("\n=== Vector Search ===");
        String query = "SELECT TOP @topK c.fileName, VectorDistance(c.textVector, @embedding) AS score FROM c WHERE VectorDistance(c.textVector, @embedding) > @threshold ORDER BY VectorDistance(c.textVector, @embedding)";

        SqlQuerySpec querySpec = new SqlQuerySpec(query, List.of(
            new SqlParameter("@embedding", embedding),
            new SqlParameter("@topK", topK),
            new SqlParameter("@threshold", threshold)
        ));

        CosmosPagedIterable<JsonNode> results = container.queryItems(querySpec, new CosmosQueryRequestOptions().setQueryMetricsEnabled(false), JsonNode.class);
        for (JsonNode item : results) {
            System.out.println(item.toPrettyString());
        }
        return results.stream()
                .map(item -> item.get("fileName").asText())
                .toList();
    }

    public static List<String> textSearch(CosmosContainer container, String searchQuery, int topK) {
        System.out.println("\n=== Text Search ===");
        List<String> terms = Arrays.asList(searchQuery.split(" "));

        // Convert to Cosmos-compatible array string
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < terms.size(); i++) {
            sb.append("\"").append(terms.get(i)).append("\"");
            if (i < terms.size() - 1) sb.append(", ");
        }
        sb.append("]");
        String termsArray = sb.toString();


        String query = "SELECT TOP @topK c.fileName FROM c ORDER BY RANK FullTextScore(c.text, " + termsArray + ")";
        SqlQuerySpec querySpec = new SqlQuerySpec(query, List.of(new SqlParameter("@topK", topK)));

        CosmosPagedIterable<JsonNode> results = container.queryItems(querySpec, new CosmosQueryRequestOptions().setQueryMetricsEnabled(false), JsonNode.class);
        for (JsonNode item : results) {
            System.out.println(item.toPrettyString());
        }
        return results.stream().map(item -> item.get("fileName").asText()).toList();
    }

    public static List<String>  hybridSearchRRF(CosmosContainer container, List<Float> embedding, String searchQuery, int topK) {
        System.out.println("\n=== Hybrid RRF Search ===");

        // Build terms array as Cosmos expects
        List<String> terms = Arrays.asList(searchQuery.split(" "));
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < terms.size(); i++) {
            sb.append("\"").append(terms.get(i)).append("\"");
            if (i < terms.size() - 1) sb.append(", ");
        }
        sb.append("]");
        String termsArray = sb.toString();

        // Build embedding array as Cosmos expects
        StringBuilder embeddingBuilder = new StringBuilder("[");
        for (int i = 0; i < embedding.size(); i++) {
            embeddingBuilder.append(embedding.get(i));
            if (i < embedding.size() - 1) embeddingBuilder.append(", ");
        }
        embeddingBuilder.append("]");
        String embeddingArray = embeddingBuilder.toString();

        // Build final SQL query string directly
        String query = String.format(
                "SELECT TOP %d c.fileName FROM c " +
                        "ORDER BY RANK RRF(FullTextScore(c.text, %s), VectorDistance(c.textVector, %s))",
                topK, termsArray, embeddingArray);

        // Execute raw query (no parameters)
        CosmosPagedIterable<JsonNode> results = container.queryItems(
                query,
                new CosmosQueryRequestOptions().setQueryMetricsEnabled(false),
                JsonNode.class
        );

        for (JsonNode item : results) {
            System.out.println(item.toPrettyString());
        }
        return results.stream().map(item -> item.get("fileName").asText()).toList();
    }
}
