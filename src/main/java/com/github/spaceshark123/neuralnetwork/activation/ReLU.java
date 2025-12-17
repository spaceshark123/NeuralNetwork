package com.github.spaceshark123.neuralnetwork.activation;

/**
 * Rectified Linear Unit activation: f(x) = max(0, x)
 * Most commonly used in hidden layers of deep networks.
 */
public class ReLU implements ActivationFunction {
    private static final long serialVersionUID = 1L;
    
    @Override
    public double activate(double raw) {
        return Math.max(0, raw);
    }
    
    @Override
    public double derivative(double raw) {
        return raw > 0 ? 1.0 : 0.0;
    }
    
    @Override
	public String getName() {
		return "ReLU";
	}
	
	@Override
	public String toConfigString() {
		return "ReLU";
	}
}