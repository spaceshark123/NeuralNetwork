package com.github.spaceshark123.neuralnetwork.loss;

/**
 * Interface for loss functions used during neural network training.
 * Loss functions measure the difference between predicted and expected outputs.
 */
public interface LossFunction {
    /**
     * Computes the loss between predicted and expected values.
     * 
     * @param predicted the predicted output values
     * @param expected the expected (target) output values
     * @return the computed loss value
     * @throws IllegalArgumentException if array lengths don't match
     */
    double compute(double[] predicted, double[] expected);
    
    /**
     * Computes the gradient of the loss with respect to all predictions.
     * This is more efficient and correct than computing individual derivatives,
     * especially for losses that depend on the entire output (like mean squared error).
     * 
     * @param predicted the predicted output values
     * @param expected the expected (target) output values
     * @return array of gradients, same length as predicted/expected
     * @throws IllegalArgumentException if array lengths don't match
     */
    double[] gradient(double[] predicted, double[] expected);
    
    /**
     * Returns the name of this loss function.
     * 
     * @return the loss function name
     */
    String getName();
    
    /**
     * Indicates whether this loss function has a special optimization when used
     * with a specific activation function (e.g., softmax + categorical crossentropy).
     * 
     * @return true if this loss function can be optimized with certain activations
     */
    default boolean hasActivationOptimization() {
        return false;
    }
    
    /**
     * Returns the activation function class that this loss function optimizes with.
     * Only called if hasActivationOptimization() returns true.
     * 
     * @return the activation function class for optimization
     */
    default Class<?> getOptimizedActivation() {
        return null;
    }
    
    /**
     * Computes the optimized gradient when used with a specific activation.
     * For example, softmax + categorical crossentropy simplifies to (predicted - expected).
     * 
     * @param predicted the predicted values
     * @param expected the expected values
     * @return array of optimized gradients
     */
    default double[] optimizedGradient(double[] predicted, double[] expected) {
        return gradient(predicted, expected);
    }
}