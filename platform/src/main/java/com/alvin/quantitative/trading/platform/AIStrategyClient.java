package com.alvin.quantitative.trading.platform;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AI Strategy Client
 * Author: Alvin
 * Communicates with Python AI service for strategy signals
 */
public class AIStrategyClient {
    private final HttpClient httpClient;
    private final String apiUrl;
    private final ObjectMapper objectMapper;
    
    public AIStrategyClient(String apiUrl) {
        this.httpClient = HttpClient.newHttpClient();
        this.apiUrl = apiUrl;
        this.objectMapper = new ObjectMapper();
    }
    
    public AISignal getSignal(String symbol, KlineData currentData, 
                             Map<String, Double> indicators, List<KlineData> history) {
        try {
            Map<String, Object> request = new HashMap<>();
            request.put("symbol", symbol);
            request.put("current_data", currentData);
            request.put("indicators", indicators);
            request.put("history", history.subList(Math.max(0, history.size() - 100), history.size()));
            
            String jsonRequest = objectMapper.writeValueAsString(request);
            
            HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(apiUrl + "/get_signal"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonRequest))
                .timeout(Duration.ofSeconds(5))
                .build();
            
            HttpResponse<String> response = httpClient.send(httpRequest, 
                HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                return objectMapper.readValue(response.body(), AISignal.class);
            } else {
                System.err.println("AI service error: " + response.statusCode());
                return createFallbackSignal("AI service unavailable");
            }
            
        } catch (Exception e) {
            System.err.println("Failed to call AI service: " + e.getMessage());
            return createFallbackSignal("Network error");
        }
    }
    
    private AISignal createFallbackSignal(String reason) {
        AISignal signal = new AISignal();
        signal.setAction("HOLD");
        signal.setConfidence(0.0);
        signal.setReason("Fallback strategy: " + reason);
        return signal;
    }
}