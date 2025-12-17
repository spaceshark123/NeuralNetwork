package com.github.spaceshark123.neuralnetwork.optimizer;

public class SGDMomentum implements Optimizer {
	private double[][] biases;
	private double[][][] weights;
	private int[] neuronsPerLayer;
	private double[][] biasVelocity;
	private double[][][] weightVelocity;
	private double momentum;

	public SGDMomentum(double momentum) {
		this.momentum = momentum;
	}

	@Override
	public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
		this.biases = biases;
		this.weights = weights;
		this.neuronsPerLayer = neuronsPerLayer;
		biasVelocity = new double[biases.length][biases[0].length];
		weightVelocity = new double[weights.length][weights[0].length][weights[0][0].length];
	}

	@Override
	public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
		for (int i = 1; i < neuronsPerLayer.length; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				// do momentum
				biasVelocity[i][j] = momentum * biasVelocity[i][j] + (1 - momentum) * avgBiasGradient[i][j];
				// apply velocity
				biases[i][j] = biases[i][j] - learningRate * biasVelocity[i][j];
				for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
					// do momentum
					weightVelocity[i][j][k] = momentum * weightVelocity[i][j][k]
							+ (1 - momentum) * avgWeightGradient[i][j][k];
					// apply velocity
					weights[i][j][k] = weights[i][j][k] - learningRate * weightVelocity[i][j][k];
				}
			}
		}
	}
}