package io.github.spaceshark123.neuralnetwork.loss;

/**
 * Binary Crossentropy loss function.
 * Used for binary classification tasks.
 * Formula: -Σ(expected * log(predicted) + (1 - expected) * log(1 - predicted))
 */
public class BinaryCrossentropy implements LossFunction {
    private static final double EPSILON = 1e-15;
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            // Clip predictions to avoid log(0)
            double p = Math.max(EPSILON, Math.min(1 - EPSILON, predicted[i]));
            sum -= expected[i] * Math.log(p) + (1 - expected[i]) * Math.log(1 - p);
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
            // Clip predictions to avoid division by zero
            double p = Math.max(EPSILON, Math.min(1 - EPSILON, predicted[i]));
            gradients[i] = (p - expected[i]) / (p * (1 - p) * n);
        }
        
        return gradients;
    }
    
    @Override
    public String getName() {
        return "BinaryCrossentropy";
    }
}