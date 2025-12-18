package io.github.spaceshark123.neuralnetwork.activation;

/**
 * Binary step activation function: f(x) = 1 if x >= 0 else 0
 * Outputs either 0 or 1.
 */
public class Binary implements ActivationFunction {
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double raw) {
		return raw >= 0.0 ? 1.0 : 0.0;
	}

	@Override
	public double derivative(double raw) {
		return 0.0; // Derivative is not defined for binary step function
	}

	@Override
	public String getName() {
		return "Binary";
	}
	
	@Override
	public String toConfigString() {
		return "Binary";
	}
}
