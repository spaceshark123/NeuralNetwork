package com.github.spaceshark123.neuralnetwork.optimizer;

public class AdaGrad implements Optimizer {
	private double[][] biases;
	private double[][][] weights;
	private int[] neuronsPerLayer;
	private double[][] biasCache;
	private double[][][] weightCache;
	private double epsilon = 1e-8;

	@Override
	public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
		this.biases = biases;
		this.weights = weights;
		this.neuronsPerLayer = neuronsPerLayer;
		biasCache = new double[biases.length][biases[0].length];
		weightCache = new double[weights.length][weights[0].length][weights[0][0].length];
	}

	@Override
	public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
		for (int i = 1; i < neuronsPerLayer.length; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				// update cache
				biasCache[i][j] += avgBiasGradient[i][j] * avgBiasGradient[i][j];
				// apply update
				biases[i][j] = biases[i][j]
						- learningRate * avgBiasGradient[i][j] / (Math.sqrt(biasCache[i][j]) + epsilon);
				for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
					// update cache
					weightCache[i][j][k] += avgWeightGradient[i][j][k] * avgWeightGradient[i][j][k];
					// apply update
					weights[i][j][k] = weights[i][j][k] - learningRate * avgWeightGradient[i][j][k]
							/ (Math.sqrt(weightCache[i][j][k]) + epsilon);
				}
			}
		}
	}
}