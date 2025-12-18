package io.github.spaceshark123.neuralnetwork.optimizer;

public class Adam implements Optimizer {
	private double[][] biases;
	private double[][][] weights;
	private int[] neuronsPerLayer;
	private double[][] biasM;
	private double[][] biasV;
	private double[][][] weightM;
	private double[][][] weightV;
	private double beta1;
	private double beta2;
	private double epsilon = 1e-8;
	private double beta1t = 1;
	private double beta2t = 1;

	public Adam(double beta1, double beta2) {
		this.beta1 = beta1;
		this.beta2 = beta2;
	}

	@Override
	public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
		this.biases = biases;
		this.weights = weights;
		this.neuronsPerLayer = neuronsPerLayer;
		biasM = new double[biases.length][biases[0].length];
		biasV = new double[biases.length][biases[0].length];
		weightM = new double[weights.length][weights[0].length][weights[0][0].length];
		weightV = new double[weights.length][weights[0].length][weights[0][0].length];
	}

	@Override
	public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
		beta1t *= beta1;
		beta2t *= beta2;
		double biasCorrectedM;
		double biasCorrectedV;
		double weightCorrectedM;
		double weightCorrectedV;
		for (int i = 1; i < neuronsPerLayer.length; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				// update biased first moment estimate
				biasM[i][j] = beta1 * biasM[i][j] + (1 - beta1) * avgBiasGradient[i][j];
				// update biased second raw moment estimate
				biasV[i][j] = beta2 * biasV[i][j] + (1 - beta2) * avgBiasGradient[i][j] * avgBiasGradient[i][j];
				// correct bias first moment
				biasCorrectedM = biasM[i][j] / (1 - beta1t);
				// correct bias second moment
				biasCorrectedV = biasV[i][j] / (1 - beta2t);
				// apply update
				biases[i][j] = biases[i][j]
						- learningRate * biasCorrectedM / (Math.sqrt(biasCorrectedV) + epsilon);
				for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
					// update biased first moment estimate
					weightM[i][j][k] = beta1 * weightM[i][j][k] + (1 - beta1) * avgWeightGradient[i][j][k];
					// update biased second raw moment estimate
					weightV[i][j][k] = beta2 * weightV[i][j][k]
							+ (1 - beta2) * avgWeightGradient[i][j][k] * avgWeightGradient[i][j][k];
					// correct bias first moment
					weightCorrectedM = weightM[i][j][k] / (1 - beta1t);
					// correct bias second moment
					weightCorrectedV = weightV[i][j][k] / (1 - beta2t);
					// apply update
					weights[i][j][k] = weights[i][j][k]
							- learningRate * weightCorrectedM / (Math.sqrt(weightCorrectedV) + epsilon);
				}
			}
		}
	}
}