package io.github.spaceshark123.neuralnetwork.loss;

/**
 * Mean Absolute Error (MAE) loss function, also known as L1 loss.
 * More robust to outliers than MSE.
 * Formula: (1/n) * Σ|predicted - expected|
 */
public class MeanAbsoluteError implements LossFunction {
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            sum += Math.abs(predicted[i] - expected[i]);
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
            if (predicted[i] > expected[i]) {
                gradients[i] = 1.0 / n;
            } else if (predicted[i] < expected[i]) {
                gradients[i] = -1.0 / n;
            } else {
                gradients[i] = 0.0; // Technically undefined at 0, but we use 0
            }
        }
        
        return gradients;
    }
    
    @Override
    public String getName() {
        return "MeanAbsoluteError";
    }
}