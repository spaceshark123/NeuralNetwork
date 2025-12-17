package com.github.spaceshark123.neuralnetwork.activation;

import java.io.Serializable;

/**
 * Interface for activation functions used in neural network layers.
 * Activation functions transform the raw weighted sum of inputs into the neuron's output.
 */
public interface ActivationFunction extends Serializable {
    /**
     * Applies the activation function to a single raw value.
     * 
     * @param raw the raw weighted sum of inputs
     * @return the activated value
     */
    public double activate(double raw);
    
    /**
     * Computes the derivative of the activation function at a given raw value.
     * Used during backpropagation.
     * 
     * @param raw the raw weighted sum of inputs
     * @return the derivative at the given point
     */
    public double derivative(double raw);
    
    /**
     * Returns the name of this activation function.
     * 
     * @return the activation function name
     */
    public String getName();

    /**
     * Returns a string representation that can be used to reconstruct this
     * activation function, including any parameters.
     * Format: "ClassName(param1=value1,param2=value2,...)"
     * 
     * @return serializable string representation
     */
    String toConfigString();
    
    /**
     * Creates an activation function from a config string.
     * 
     * @param config the configuration string
     * @return the activation function instance
     * @throws IllegalArgumentException if the config string is invalid
     */
    static ActivationFunction fromConfigString(String config) {
        return ActivationFunctionFactory.create(config);
    }
    
    /**
     * Indicates whether this activation function requires access to all neurons
     * in the layer (like softmax) rather than operating on individual neurons.
     * 
     * @return true if the function needs the full layer context
     */
    public default boolean requiresLayerContext() {
        return false;
    }
    
    /**
     * Applies the activation function with full layer context.
     * Only called if requiresLayerContext() returns true.
     * 
     * @param raw the raw value for this specific neuron
     * @param allRawValues all raw values in the layer
     * @param index the index of the current neuron
     * @return the activated value
     */
    default double activateWithContext(double raw, double[] allRawValues, int index) {
        return activate(raw);
    }
    
    /**
     * Computes the derivative with full layer context.
     * Only called if requiresLayerContext() returns true.
     * 
     * @param raw the raw value for this specific neuron
     * @param allRawValues all raw values in the layer
     * @param index the index of the current neuron
     * @return the derivative
     */
    default double derivativeWithContext(double raw, double[] allRawValues, int index) {
        return derivative(raw);
    }
}