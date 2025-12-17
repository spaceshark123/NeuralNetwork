package com.github.spaceshark123.neuralnetwork.activation;

/**
 * Softmax activation function: converts raw values into a probability distribution.
 * Typically used in the output layer for multi-class classification.
 * 
 * <p>Note: This activation function requires access to all neurons in the layer
 * (requiresLayerContext() returns true).
 */
public class Softmax implements ActivationFunction {
    private static final long serialVersionUID = 1L;
    
    @Override
    public double activate(double raw) {
        throw new UnsupportedOperationException(
            "Softmax requires layer context. Use activateWithContext() instead.");
    }
    
    @Override
    public double derivative(double raw) {
        throw new UnsupportedOperationException(
            "Softmax requires layer context. Use derivativeWithContext() instead.");
    }
    
    @Override
    public boolean requiresLayerContext() {
        return true;
    }
    
    @Override
    public double activateWithContext(double raw, double[] allRawValues, int index) {
        // Numerical stability: subtract max value
        double maxVal = max(allRawValues);
        
        // Compute the normalization factor (sum of exponentials)
        double total = 0;
		for (double value : allRawValues) {
			total += Math.exp(value - maxVal);
		}
		
		// Compute the softmax activation
		return Math.exp(raw - maxVal - Math.log(total));
    }
    
    @Override
    public double derivativeWithContext(double raw, double[] allRawValues, int index) {
        double softmax = activateWithContext(raw, allRawValues, index);
        return softmax * (1.0 - softmax);
    }
    
    @Override
	public String getName() {
		return "Softmax";
	}
	
	@Override
	public String toConfigString() {
		return "Softmax";
	}
    
    private double max(double[] arr) {
        double max = arr[0];
        for (double val : arr) {
            if (val > max) max = val;
        }
        return max;
    }
}