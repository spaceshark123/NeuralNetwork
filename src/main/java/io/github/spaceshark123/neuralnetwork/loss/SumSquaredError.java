package io.github.spaceshark123.neuralnetwork.loss;

/**
 * Sum of Squared Errors (SSE) loss function.
 * Similar to MSE but without the averaging.
 * Formula: 0.5 * Σ(predicted - expected)²
 */
public class SumSquaredError implements LossFunction {
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException(
                "Predicted and expected arrays must have the same length");
        }
        
        double sum = 0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = expected[i] - predicted[i];
            sum += 0.5 * diff * diff;
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
            gradients[i] = predicted[i] - expected[i];
        }
        
        return gradients;
    }
    
    @Override
    public String getName() {
        return "SumSquaredError";
    }
}