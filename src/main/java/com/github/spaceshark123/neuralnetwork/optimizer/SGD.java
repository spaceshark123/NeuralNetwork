package com.github.spaceshark123.neuralnetwork.optimizer;

public class SGD implements Optimizer {
	private double[][] biases;
	private double[][][] weights;
	private int[] neuronsPerLayer;

	@Override
	public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
		this.biases = biases;
		this.weights = weights;
		this.neuronsPerLayer = neuronsPerLayer;
	}

	@Override
	public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
		for (int i = 1; i < neuronsPerLayer.length; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				// apply velocity
				biases[i][j] = biases[i][j] - learningRate * avgBiasGradient[i][j];
				for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
					// apply velocity
					weights[i][j][k] = weights[i][j][k] - learningRate * avgWeightGradient[i][j][k];
				}
			}
		}
	}
}