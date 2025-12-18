package io.github.spaceshark123.neuralnetwork.activation;

/**
 * Linear activation function: f(x) = x
 * No transformation applied to the input.
 */
public class Linear implements ActivationFunction {
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double raw) {
		return raw;
	}

	@Override
	public double derivative(double raw) {
		return 1.0;
	}

	@Override
	public String getName() {
		return "Linear";
	}

	@Override
	public String toConfigString() {
		return "Linear";
	}
}