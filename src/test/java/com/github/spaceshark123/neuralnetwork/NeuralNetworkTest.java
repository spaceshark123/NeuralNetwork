package com.github.spaceshark123.neuralnetwork;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

import com.github.spaceshark123.neuralnetwork.activation.*;
import com.github.spaceshark123.neuralnetwork.loss.*;
import com.github.spaceshark123.neuralnetwork.optimizer.*;

public class NeuralNetworkTest {

	@Test
	public void testNeuralNetworkCreation() {
		int[] topology = { 3, 5, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		Assertions.assertNotNull(nn);
		Assertions.assertEquals(3, nn.getTopology().length);
		Assertions.assertEquals(5, nn.getTopology()[1]);
	}

	@Test
	public void testInvalidNeuralNetworkCreation() {
		int[] topology = { 3, 5, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear()
		};
		Assertions.assertThrows(IllegalArgumentException.class, () -> {
			new NeuralNetwork(topology, activations);
		});
	}

	@Test
	public void testNeuralNetworkInit() {
		int[] topology = { 4, 6, 3 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		nn.init(1); // init with bias spread of 1
		double[][][] weights = nn.getWeights();
		double[][] biases = nn.getBiases();
		// weights should now be non zero
		boolean nonZeroWeights = false;
		for (int i = 1; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					if (weights[i][j][k] != 0) {
						nonZeroWeights = true;
						break;
					}
				}
				if (nonZeroWeights) {
					break;
				}
			}
			if (nonZeroWeights) {
				break;
			}
		}
		Assertions.assertTrue(nonZeroWeights);
		// biases should now be non zero
		boolean nonZeroBiases = false;
		for (int i = 1; i < biases.length; i++) {
			for (int j = 0; j < biases[i].length; j++) {
				if (biases[i][j] != 0) {
					nonZeroBiases = true;
					break;
				}
			}
			if (nonZeroBiases) {
				break;
			}
		}
		Assertions.assertTrue(nonZeroBiases);
	}

	@Test
	public void testNeuralNetworkPredict() {
		int[] topology = { 2, 4, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		nn.init();
		double[] input = { 0.5, 0.8 };
		double[] output = nn.evaluate(input);
		Assertions.assertNotNull(output);
		Assertions.assertEquals(2, output.length);
	}

	@Test
	public void testNeuralNetworkString() {
		int[] topology = { 2, 3, 1 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Sigmoid()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		String nnString = nn.toString();
		Assertions.assertNotNull(nnString);
	}

	@Test
	public void testNeuralNetworkGetters() {
		int[] topology = { 3, 5, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		Assertions.assertArrayEquals(topology, nn.getTopology());
		Assertions.assertArrayEquals(activations, nn.getActivations());
	}

	@Test
	public void testNeuralNetworkClone() {
		int[] topology = { 3, 5, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		NeuralNetwork nnClone = nn.clone();
		Assertions.assertNotSame(nn, nnClone);
		Assertions.assertArrayEquals(nn.getTopology(), nnClone.getTopology());
		Assertions.assertArrayEquals(nn.getActivations(), nnClone.getActivations());
	}

	@Test
	public void testNeuralNetworkTrain() {
		int[] topology = { 2, 4, 2 };
		ActivationFunction[] activations = {
				new ReLU(),
				new Linear(),
				new Softmax()
		};
		NeuralNetwork nn = new NeuralNetwork(topology, activations);
		nn.init();
		// Simple dataset for testing
		double[][] train_X = {
				{ 0.1, 0.2 },
				{ 0.3, 0.4 },
				{ 0.5, 0.6 },
				{ 0.7, 0.8 }
		};
		double[][] train_y = {
				{ 1.0, 0.0 },
				{ 0.0, 1.0 },
				{ 1.0, 0.0 },
				{ 0.0, 1.0 }
		};
		double[][] test_X = {
				{ 0.15, 0.25 },
				{ 0.35, 0.45 }
		};
		double[][] test_y = {
				{ 1.0, 0.0 },
				{ 0.0, 1.0 }
		};
		Optimizer optimizer = new SGD();
		// evaluate before training
		double correctBefore = 0;
		for (int i = 0; i < train_X.length; i++) {
			double[] output = nn.evaluate(train_X[i]);
			int predicted = output[0] > output[1] ? 0 : 1;
			int actual = train_y[i][0] > train_y[i][1] ? 0 : 1;
			if (predicted == actual) {
				correctBefore++;
			}
		}
		nn.train(train_X, train_y, test_X, test_y, 10, 0.01, 2, new CategoricalCrossentropy(), optimizer);
		// After training, evaluate on test set
		double correct = 0;
		for (int i = 0; i < train_X.length; i++) {
			double[] output = nn.evaluate(train_X[i]);
			int predicted = output[0] > output[1] ? 0 : 1;
			int actual = train_y[i][0] > train_y[i][1] ? 0 : 1;
			if (predicted == actual) {
				correct++;
			}
		}
		Assertions.assertTrue(correct >= correctBefore); // accuracy should not decrease
	}
}