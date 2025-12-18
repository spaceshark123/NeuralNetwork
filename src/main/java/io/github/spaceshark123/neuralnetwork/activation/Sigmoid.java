package io.github.spaceshark123.neuralnetwork.activation;

/**
 * Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * Squashes values to range (0, 1).
 */
public class Sigmoid implements ActivationFunction {
    private static final long serialVersionUID = 1L;
    
    @Override
    public double activate(double raw) {
        return 1.0 / (1.0 + Math.exp(-raw));
    }
    
    @Override
    public double derivative(double raw) {
        double sigmoid = activate(raw);
        return sigmoid * (1.0 - sigmoid);
    }
    
    @Override
	public String getName() {
		return "Sigmoid";
	}
	
	@Override
	public String toConfigString() {
		return "Sigmoid";
	}
}