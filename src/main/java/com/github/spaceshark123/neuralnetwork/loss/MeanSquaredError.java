package com.github.spaceshark123.neuralnetwork.loss;

/**
 * Mean Squared Error (MSE) loss function.
 * Commonly used for regression tasks.
 * Formula: (1/n) * Σ(predicted - expected)²
 */
public class MeanSquaredError implements LossFunction {
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - expected[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }
    
    @Override
    public double[] gradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double[] gradients = new double[predicted.length];
        int n = predicted.length;
        
        for (int i = 0; i < n; i++) {
            gradients[i] = 2.0 * (predicted[i] - expected[i]) / n;
        }
        
        return gradients;
    }
    
    @Override
    public String getName() {
        return "MeanSquaredError";
    }
}