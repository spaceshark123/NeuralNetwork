package io.github.spaceshark123.neuralnetwork.loss;

import io.github.spaceshark123.neuralnetwork.activation.Softmax;

/**
 * Categorical Crossentropy loss function.
 * Used for multi-class classification with one-hot encoded labels.
 * Formula: -Σ(expected * log(predicted))
 * 
 * <p>When used with Softmax activation in the output layer, the gradient
 * simplifies to (predicted - expected), which significantly speeds up computation.
 */
public class CategoricalCrossentropy implements LossFunction {
    private static final double EPSILON = 1e-15;
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            // Add epsilon for numerical stability (avoid log(0))
            sum -= expected[i] * Math.log(predicted[i] + EPSILON);
        }
        return sum;
    }
    
    @Override
    public double[] gradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double[] gradients = new double[predicted.length];
        
        for (int i = 0; i < predicted.length; i++) {
            // Add epsilon for numerical stability
            gradients[i] = -expected[i] / (predicted[i] + EPSILON);
        }
        
        return gradients;
    }
    
    @Override
    public String getName() {
        return "CategoricalCrossentropy";
    }
    
    @Override
    public boolean hasActivationOptimization() {
        return true;
    }
    
    @Override
    public Class<?> getOptimizedActivation() {
        return Softmax.class;
    }
    
    @Override
    public double[] optimizedGradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        // When used with softmax, the gradient simplifies dramatically
        double[] gradients = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            gradients[i] = predicted[i] - expected[i];
        }
        return gradients;
    }
}