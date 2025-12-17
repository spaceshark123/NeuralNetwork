package com.github.spaceshark123.neuralnetwork.activation;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Factory for creating activation functions from string configurations.
 * Supports both simple names (case-insensitive) and parameterized forms.
 */
public class ActivationFunctionFactory {
    
    // Pattern to parse "FunctionName(param1=value1,param2=value2)"
    private static final Pattern PARAM_PATTERN = Pattern.compile("([^(]+)\\((.*)\\)");
    private static final Pattern KEY_VALUE_PATTERN = Pattern.compile("([^=,]+)=([^=,]+)");
    
    // Registry of activation function creators
    private static final Map<String, ActivationCreator> REGISTRY = new HashMap<>();
    
    static {
        // Register built-in activation functions (case-insensitive)
        register("linear", params -> new Linear());
        register("sigmoid", params -> new Sigmoid());
        register("tanh", params -> new Tanh());
        register("relu", params -> new ReLU());
        register("softmax", params -> new Softmax());
        register("binary", params -> new Binary());
        
        // Parameterized activation functions
        register("leakyrelu", params -> {
            double alpha = parseDouble(params, "alpha", 0.01);
            return new LeakyReLU(alpha);
        });
        
        // Add more as needed...
    }
    
    /**
     * Creates an activation function from a configuration string.
     * 
     * @param config configuration string (e.g., "ReLU", "leakyrelu", "LeakyReLU(alpha=0.2)")
     * @return the activation function
     * @throws IllegalArgumentException if config is invalid
     */
    public static ActivationFunction create(String config) {
        if (config == null || config.trim().isEmpty()) {
            throw new IllegalArgumentException("Config string cannot be null or empty");
        }
        
        config = config.trim();
        
        // Check if it has parameters
        Matcher matcher = PARAM_PATTERN.matcher(config);
        
        if (matcher.matches()) {
            // Has parameters: "FunctionName(param1=value1,param2=value2)"
            String name = matcher.group(1).trim().toLowerCase();
            String paramsStr = matcher.group(2).trim();
            
            Map<String, String> params = parseParameters(paramsStr);
            
            ActivationCreator creator = REGISTRY.get(name);
            if (creator == null) {
                throw new IllegalArgumentException("Unknown activation function: " + name);
            }
            
            return creator.create(params);
        } else {
            // No parameters: just the name
            String name = config.toLowerCase();
            
            ActivationCreator creator = REGISTRY.get(name);
            if (creator == null) {
                throw new IllegalArgumentException("Unknown activation function: " + name);
            }
            
            return creator.create(new HashMap<>());
        }
    }
    
    /**
     * Registers a custom activation function creator.
     * Allows users to add their own activation functions to the factory.
     * 
     * @param name the name (case-insensitive)
     * @param creator the creator function
     */
    public static void register(String name, ActivationCreator creator) {
        REGISTRY.put(name.toLowerCase(), creator);
    }
    
    /**
     * Parses parameter string into a map.
     * Example: "alpha=0.01,beta=0.5" -> {"alpha": "0.01", "beta": "0.5"}
     */
    private static Map<String, String> parseParameters(String paramsStr) {
        Map<String, String> params = new HashMap<>();
        
        if (paramsStr.isEmpty()) {
            return params;
        }
        
        Matcher matcher = KEY_VALUE_PATTERN.matcher(paramsStr);
        while (matcher.find()) {
            String key = matcher.group(1).trim();
            String value = matcher.group(2).trim();
            params.put(key, value);
        }
        
        return params;
    }
    
    /**
     * Helper to parse a double parameter with a default value.
     */
    private static double parseDouble(Map<String, String> params, String key, double defaultValue) {
        String value = params.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid double value for parameter '" + key + "': " + value);
        }
    }
    
    /**
     * Helper to parse an int parameter with a default value.
     */
    private static int parseInt(Map<String, String> params, String key, int defaultValue) {
        String value = params.get(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid int value for parameter '" + key + "': " + value);
        }
    }
    
    /**
     * Functional interface for creating activation functions.
     */
    @FunctionalInterface
    public interface ActivationCreator {
        ActivationFunction create(Map<String, String> params);
    }
}