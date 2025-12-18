package io.github.spaceshark123.neuralnetwork.activation;

/**
 * Hyperbolic tangent activation function: f(x) = tanh(x)
 * Squashes values to range (-1, 1).
 */
public class Tanh implements ActivationFunction {
	private static final long serialVersionUID = 1L;

	@Override
	public double activate(double raw) {
		return Math.tanh(raw);
	}

	@Override
	public double derivative(double raw) {
		return Math.pow(1.0 / Math.cosh(raw), 2);
	}

	@Override
	public String getName() {
		return "Tanh";
	}

	@Override
	public String toConfigString() {
		return "Tanh";
	}
}