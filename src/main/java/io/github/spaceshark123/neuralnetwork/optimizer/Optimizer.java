package io.github.spaceshark123.neuralnetwork.optimizer;

/**
 * Interface for optimization algorithms used in neural network training.
 * Optimizers maintain internal state and update network parameters based on gradients.
 */
public interface Optimizer {
    /**
     * Initializes the optimizer with the network's structure and parameters.
     * 
     * @param neuronsPerLayer number of neurons in each layer
     * @param biases the network's bias arrays (will be modified during training)
     * @param weights the network's weight arrays (will be modified during training)
     */
    void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights);
    
    /**
     * Performs a single optimization step, updating weights and biases.
     * 
     * @param avgBiasGradient average gradient for biases across the batch
     * @param avgWeightGradient average gradient for weights across the batch
     * @param learningRate current learning rate
     */
    void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate);
}