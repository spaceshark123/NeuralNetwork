package io.github.spaceshark123.neuralnetwork.activation;

/**
 * Leaky ReLU activation: f(x) = max(alpha*x, x)
 * Similar to ReLU but allows small negative values.
 */
public class LeakyReLU implements ActivationFunction {
	private static final long serialVersionUID = 1L;
	private final double alpha;

	/**
	 * Creates a Leaky ReLU with default alpha = 0.01
	 */
	public LeakyReLU() {
		this(0.01);
	}

	/**
	 * Creates a Leaky ReLU with specified alpha
	 * 
	 * @param alpha the slope for negative values
	 */
	public LeakyReLU(double alpha) {
		this.alpha = alpha;
	}

	@Override
	public double activate(double raw) {
		return raw > 0 ? raw : alpha * raw;
	}

	@Override
	public double derivative(double raw) {
		return raw > 0 ? 1.0 : alpha;
	}

	@Override
	public String getName() {
		return "LeakyReLU(alpha=" + alpha + ")";
	}

	@Override
	public String toConfigString() {
		return "LeakyReLU(alpha=" + alpha + ")";
	}
}