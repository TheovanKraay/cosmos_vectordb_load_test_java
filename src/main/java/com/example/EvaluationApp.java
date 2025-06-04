package com.example;

import com.azure.ai.openai.OpenAIAsyncClient;
import com.azure.ai.openai.OpenAIClientBuilder;
import com.azure.ai.openai.models.EmbeddingItem;
import com.azure.ai.openai.models.EmbeddingsOptions;
import com.azure.core.credential.AzureKeyCredential;
import com.azure.core.http.policy.ExponentialBackoffOptions;
import com.azure.core.http.policy.RetryOptions;
import com.azure.cosmos.*;
import com.azure.identity.DefaultAzureCredentialBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.*;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

public class EvaluationApp {

    private static final String COSMOS_ENDPOINT = System.getenv("COSMOS_ENDPOINT");
    private static final String DATABASE_NAME = "books";
    private static final String CONTAINER_NAME = "bookitems4";
    private static final String OPENAI_API_KEY = System.getenv("AZURE_OPENAI_API_KEY");
    private static final String OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002";
    private static final String OPENAI_EMBEDDING_URL = System.getenv("OPENAI_ENDPOINT");

    private static final ObjectMapper mapper = new ObjectMapper();
    private static OpenAIAsyncClient openAIClient;
    private static CosmosContainer container;

    public static void main(String[] args) throws Exception {
        RetryOptions retryOptions = new RetryOptions(
                new ExponentialBackoffOptions()
                        .setMaxRetries(10)
                        .setMaxDelay(Duration.of(2, ChronoUnit.SECONDS)));

        openAIClient = new OpenAIClientBuilder()
                .endpoint(OPENAI_EMBEDDING_URL)
                .credential(new AzureKeyCredential(OPENAI_API_KEY))
                .retryOptions(retryOptions)
                .buildAsyncClient();

        CosmosClient client = new CosmosClientBuilder()
                .endpoint(COSMOS_ENDPOINT)
                .credential(new DefaultAzureCredentialBuilder().build())
                .buildClient();

        container = client.getDatabase(DATABASE_NAME).getContainer(CONTAINER_NAME);

        // Hardcoded questions
        List<String> questions = List.of(
                "Give me a list of books published in the year 2000",
                "Give me a list of book of travel category",
                "Give me a list of books from author Agatha Christie",
                "Give me some haunted incidents from california state",
                "Give me some wines found in Italy",
                "Give me wines tasted by Roger Voss",
                "Give me some wines in the variety of Red Blend",
                "Give me some business category news",
                "Give me a list of students graduated in the year 2025"
        );

        double totalVectorRecall = 0;
        double totalTextRecall = 0;
        double totalHybridRecall = 0;

        for (int i = 0; i < questions.size(); i++) {
            String query = questions.get(i);
            String questionFile = String.format("books_processing/questions/question%d.txt", i + 1);
            List<String> groundTruthFiles = loadGroundTruthFile(questionFile);

            for (int j = 0; j < groundTruthFiles.size(); j++) {
                System.out.println("Ground Truth File " + (j + 1) + ": " + groundTruthFiles.get(j));
            }

            if (groundTruthFiles.isEmpty()) {
                System.out.println("\nSkipping query with missing ground truth: " + questionFile);
                continue;
            }

            System.out.println("\nEvaluating question " + (i+1));
            List<Float> embedding = getEmbedding(query);

            List<String> vectorResults = VectorSearchApp.vectorSearch(container, embedding, 5, 0.7);
            Thread.sleep(500);
            List<String> textResults = VectorSearchApp.textSearch(container, query, 5);
            Thread.sleep(500);
            List<String> hybridResults = VectorSearchApp.hybridSearchRRF(container, embedding, query, 5);
            Thread.sleep(500);

            double vectorRecall = computeRecall(vectorResults, groundTruthFiles);
            double textRecall = computeRecall(textResults, groundTruthFiles);
            double hybridRecall = computeRecall(hybridResults, groundTruthFiles);

            totalVectorRecall += vectorRecall;
            totalTextRecall += textRecall;
            totalHybridRecall += hybridRecall;

            System.out.printf("Vector Recall@5: %.2f | Text Recall@5: %.2f | Hybrid Recall@5: %.2f%n",
                    vectorRecall, textRecall, hybridRecall);
        }

        int evaluatedQueries = questions.size();
        System.out.printf("\nOverall Average Recall@5 --> Vector: %.2f | Text: %.2f | Hybrid: %.2f%n",
                totalVectorRecall / evaluatedQueries,
                totalTextRecall / evaluatedQueries,
                totalHybridRecall / evaluatedQueries);

        client.close();
    }

    private static String normalizeFileName(String fileName) {
        fileName = fileName.trim();
/*        if (fileName.startsWith("book_")) {
            fileName = fileName.substring(5);
        }*/
/*        if (fileName.endsWith(".txt") || fileName.endsWith(".csv")) {
            fileName = fileName.substring(0, fileName.lastIndexOf('.'));
        }*/
        return fileName;
    }


    private static List<String> loadGroundTruthFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) return Collections.emptyList();

        return Files.readAllLines(path).stream()
                .map(EvaluationApp::normalizeFileName)
                .collect(Collectors.toList());
    }



    private static List<Float> getEmbedding(String query) {
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

    private static double computeRecall(List<String> results, List<String> groundTruth) {
        long hits = results.stream().filter(groundTruth::contains).count();
        System.out.printf("Hits: %d, Total Ground Truth: %d%n", hits, groundTruth.size());
        return (double) hits / groundTruth.size();
    }
}
