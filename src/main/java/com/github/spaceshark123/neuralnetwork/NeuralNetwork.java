package com.github.spaceshark123.neuralnetwork;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Collections;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.FileReader;

import java.util.stream.IntStream;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

import com.github.spaceshark123.neuralnetwork.callback.TrainingCallback;
import com.github.spaceshark123.neuralnetwork.optimizer.Optimizer;
import com.github.spaceshark123.neuralnetwork.optimizer.SGD;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;

	// first dimension is layer, second dimension is neuron # in layer
	protected double[][] neurons;
	protected double[][] neuronsRaw;
	protected double[][] biases;
	// first dimension is recieving layer, second dimension is recieving neuron #,
	// third dimension is incoming neuron # from previous layer
	protected double[][][] weights;
	protected int[] neuronsPerLayer;
	/*
	 * activation choices:
	 * - linear
	 * - sigmoid
	 * - tanh
	 * - relu
	 * - binary
	 * - softmax
	 */
	protected String[] activations;
	public int numLayers;
	// gradient clipping threshold
	public double clipThreshold = 1;
	// whether or not to display accuracy while training (for classification models)
	public boolean displayAccuracy = false;

	// Regularization type
	public static enum RegularizationType {
		NONE,
		L1,
		L2
	}

	// Regularization settings (lambda = regularization strength)
	protected double lambda = 0;
	protected RegularizationType regularizationType = RegularizationType.NONE;
	// used in gradient descent
	volatile protected double[][] avgBiasGradient;
	volatile protected double[][][] avgWeightGradient;

	// takes in int[] for number of neurons in each layer and string[] for
	// activations of each layer
	public NeuralNetwork(int[] topology, String[] active) {
		int maxLayerSize = max(topology);
		neuronsPerLayer = topology.clone();
		numLayers = topology.length;
		neurons = new double[numLayers][maxLayerSize];
		neuronsRaw = new double[numLayers][maxLayerSize];
		biases = new double[numLayers][maxLayerSize];
		weights = new double[numLayers][maxLayerSize][maxLayerSize];
		activations = active.clone();
	}

	public NeuralNetwork(int[] topology, String[] active, RegularizationType regularizationType,
			double regularizationStrength) {
		this(topology, active);
		// set regularization
		this.regularizationType = regularizationType;
		lambda = regularizationStrength;
	}

	public NeuralNetwork() {

	}

	// initialize network with random starting values
	public void init(double biasSpread) {
		clearNeurons();
		initWeights();
		initBiases(biasSpread);
	}

	// initialize network with random starting values using a specified weight
	// initialization method ('he' or 'xavier')
	public void init(String weightInitMethod, double biasSpread) {
		clearNeurons();
		initWeights(weightInitMethod);
		initBiases(biasSpread);
	}

	protected void initWeights(String initMethod) {
		// initMethod is either "he" or "xavier"
		if (initMethod.equals("he")) {
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				// he weight initialization (for relu) (gaussian distribution)
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = ThreadLocalRandom.current().nextGaussian() * std + mean;
					}
				}

			}
		} else if (initMethod.equals("xavier")) {
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				double min, max;
				// xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform
				// distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		} else {
			initWeights();
		}
	}

	protected void initWeights() {
		for (int i = 1; i < numLayers; i++) {
			int n = neuronsPerLayer[i - 1];
			double min, max;
			if (activations[i].equals("relu")) {
				// he weight initialization (for relu) (gaussian distribution)
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = ThreadLocalRandom.current().nextGaussian() * std + mean;
					}
				}
			} else {
				// xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform
				// distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		}
	}

	protected void initBiases(double spread) {
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				biases[i][j] = randDouble(-spread, spread);
			}
		}
	}

	public double[][][] getWeights() {
		return deepCopy(weights);
	}

	public void setWeight(int layer, int outgoing, int incoming, double value) {
		weights[layer][outgoing][incoming] = value;
	}

	public double[][] getBiases() {
		return deepCopy(biases);
	}

	public void setBias(int layer, int neuron, double bias) {
		biases[layer][neuron] = bias;
	}

	public String[] getActivations() {
		return activations.clone();
	}

	public void setActivation(int layer, String act) {
		activations[layer] = act;
	}

	public double[][] getNeurons() {
		return deepCopy(neurons);
	}

	public int[] getTopology() {
		return neuronsPerLayer.clone();
	}

	public void setRegularizationType(RegularizationType regularizationType) {
		this.regularizationType = regularizationType;
	}

	public void setRegularizationLambda(double lambda) {
		this.lambda = lambda;
	}

	public RegularizationType getRegularizationType() {
		return regularizationType;
	}

	public double getRegularizationLambda() {
		return lambda;
	}

	private double randDouble(double min, double max) {
		return min + (max - min) * ThreadLocalRandom.current().nextDouble();
	}

	// Total regularization term calculation
	double regularizationTerm() {
		double regTerm = 0.0;
		if (regularizationType == RegularizationType.L1) {
			// L1 regularization
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						regTerm += Math.abs(weights[i][j][k]);
					}
				}
			}
		} else if (regularizationType == RegularizationType.L2) {
			// L2 regularization
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						regTerm += weights[i][j][k] * weights[i][j][k];
					}
				}
			}
		}
		regTerm *= lambda;
		return regTerm;
	}

	protected double linear_activation(double raw) {
		return raw;
	}

	protected double sigmoid_activation(double raw) {
		return 1d / (1 + Math.exp(-raw));
	}

	protected double tanh_activation(double raw) {
		return Math.tanh(raw);
	}

	protected double relu_activation(double raw) {
		return Math.max(0, raw);
	}

	protected double binary_activation(double raw) {
		return raw > 0 ? 1 : 0;
	}

	protected double softmax_activation(double raw, double[] neuronValues) {
		double maxVal = max(neuronValues);

		// Compute the normalization factor (sum of exponentials)
		double total = 0;
		for (double value : neuronValues) {
			total += Math.exp(value - maxVal);
		}
		// Compute the softmax activation
		return Math.exp(raw - maxVal - Math.log(total));
	}

	protected double softmax_der(double[] neuronValues, int index) {
		double softmax = softmax_activation(neuronValues[index], neuronValues);
		return softmax * (1.0 - softmax);
	}

	protected double activate(double raw, int layer) {
		switch (activations[layer]) {
			case "linear":
				return linear_activation(raw);
			case "sigmoid":
				return sigmoid_activation(raw);
			case "tanh":
				return tanh_activation(raw);
			case "relu":
				return relu_activation(raw);
			case "binary":
				return binary_activation(raw);
			case "softmax":
				return softmax_activation(raw, Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]));
			default:
				return linear_activation(raw);
		}
	}

	protected double activate(double raw, int layer, double[] neuronsRaw) {
		switch (activations[layer]) {
			case "linear":
				return linear_activation(raw);
			case "sigmoid":
				return sigmoid_activation(raw);
			case "tanh":
				return tanh_activation(raw);
			case "relu":
				return relu_activation(raw);
			case "binary":
				return binary_activation(raw);
			case "softmax":
				return softmax_activation(raw, neuronsRaw);
			default:
				return linear_activation(raw);
		}
	}

	protected double activate_der(double raw, int layer, int index) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				return val;
			case "relu":
				if (raw <= 0) {
					return 0;
				} else {
					return 1;
				}
			case "binary":
				return 0;
			case "softmax":
				val = softmax_der(Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]), index);
				return val;
			default:
				return 1;
		}
	}

	protected double activate_der(double raw, int layer, double[] neuronsRaw, int index) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				return val;
			case "relu":
				if (raw <= 0) {
					return 0;
				} else {
					return 1;
				}
			case "binary":
				return 0;
			case "softmax":
				val = softmax_der(neuronsRaw, index);
				return val;
			default:
				return 1;
		}
	}

	protected void clearNeurons() {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	protected void clearNeurons(double[][] neurons, double[][] neuronsRaw) {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	public double[] evaluate(double[] input, double[][] neurons, double[][] neuronsRaw) {
		clearNeurons(neurons, neuronsRaw);

		// Set input neurons
		IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

		// Feed forward
		for (int layer = 1; layer < numLayers; layer++) {
			final int currentLayer = layer; // Capture the current value of layer
			IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(neuron -> {
				double raw = biases[currentLayer][neuron];
				for (int prevNeuron = 0; prevNeuron < neuronsPerLayer[currentLayer - 1]; prevNeuron++) {
					raw += weights[currentLayer][neuron][prevNeuron] * neurons[currentLayer - 1][prevNeuron];
				}
				neuronsRaw[currentLayer][neuron] = raw;
				if (activations[currentLayer].equals("softmax")) {
					neurons[currentLayer][neuron] = raw;
				} else {
					neurons[currentLayer][neuron] = activate(raw, currentLayer);
				}
			});

			if (activations[currentLayer].equals("softmax")) {
				IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
					neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer,
							Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]));
				});
			}
		}

		// return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	public double[] evaluate(double[] input) {
		clearNeurons();

		// Set input neurons
		IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

		// Feed forward
		for (int layer = 1; layer < numLayers; layer++) {
			final int currentLayer = layer; // Capture the current value of layer
			IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(neuron -> {
				double raw = biases[currentLayer][neuron];
				for (int prevNeuron = 0; prevNeuron < neuronsPerLayer[currentLayer - 1]; prevNeuron++) {
					raw += weights[currentLayer][neuron][prevNeuron] * neurons[currentLayer - 1][prevNeuron];
				}
				neuronsRaw[currentLayer][neuron] = raw;
				if (activations[currentLayer].equals("softmax")) {
					neurons[currentLayer][neuron] = raw;
				} else {
					neurons[currentLayer][neuron] = activate(raw, currentLayer);
				}
			});

			if (activations[currentLayer].equals("softmax")) {
				IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
					neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer);
				});
			}
		}

		// return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	@Override
	public String toString() {
		StringBuilder print = new StringBuilder().append("Neural Network \n");
		print.append("\nTopology (neurons per layer): ").append(printArr(neuronsPerLayer));
		print.append("\nActivations (per layer): ").append(printArr(activations));
		print.append("\nRegularization: ").append(regularizationType.toString()).append(" lambda: ").append(lambda);

		print.append("\nBiases:\n");
		for (int i = 0; i < numLayers; i++) {
			print.append("Layer ").append((i + 1)).append(": ")
					.append(printArr(Arrays.copyOfRange(biases[i], 0, neuronsPerLayer[i]))).append("\n");
		}

		print.append("\nWeights:\n");
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				// each neuron
				print.append("    Neuron ").append((j + 1)).append(" of Layer ").append((i + 1)).append(" Weights: \n")
						.append(printArr(Arrays.copyOfRange(weights[i][j], 0, neuronsPerLayer[i - 1]))).append("\n");
			}
		}
		return print.toString();
	}

	private String printArr(int[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	private String printArr(double[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	private String printArr(String[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		StringBuilder print = new StringBuilder().append("[");
		for (int i = 0; i < arr.length - 1; i++) {
			print.append(arr[i]).append(", ");
		}
		print.append(arr[arr.length - 1]).append("]");
		return print.toString();
	}

	// save the neural network to a file directly as a java object
	// not transferable between different programming languages and not human
	// readable
	public static void save(NeuralNetwork network, String path) {
		try {
			FileOutputStream f = new FileOutputStream(path);
			ObjectOutputStream o = new ObjectOutputStream(f);

			// Write objects to file
			o.writeObject(network);

			o.close();
			f.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// save the neural network to a file as a plain text file
	// transferable between different programming languages and human readable
	public static void saveParameters(NeuralNetwork network, String path) {
		BufferedWriter writer = null;
		try {
			FileWriter fWriter = new FileWriter(path);
			writer = new BufferedWriter(fWriter);
			StringBuilder print = new StringBuilder();
			// write parameters to print
			print.append("numlayers ").append(network.numLayers).append("\n");
			print.append("topology ");
			for (int i = 0; i < network.neuronsPerLayer.length; i++) {
				print.append(network.neuronsPerLayer[i]).append(" ");
			}
			print.append("\nactivations ");
			for (int i = 0; i < network.activations.length; i++) {
				print.append(network.activations[i]).append(" ");
			}
			print.append("\nregularization ").append(network.regularizationType.toString()).append(" ")
					.append(network.lambda).append("\n");
			print.append("biases ");
			for (int i = 0; i < network.biases.length; i++) {
				for (int j = 0; j < network.neuronsPerLayer[i]; j++) {
					print.append(network.biases[i][j]).append(" ");
				}
			}
			// weights start at layer 1 because layer 0 is the input layer
			print.append("\nweights ");
			for (int i = 1; i < network.weights.length; i++) {
				for (int j = 0; j < network.neuronsPerLayer[i]; j++) {
					for (int k = 0; k < network.neuronsPerLayer[i - 1]; k++) {
						print.append(network.weights[i][j][k]).append(" ");
					}
				}
			}
			// write to file
			writer.write(print.toString());
			writer.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// load a neural network from a file that was saved directly as a java object
	// not transferable between different programming languages
	public static NeuralNetwork load(String path) {
		try {
			FileInputStream fi = new FileInputStream(path);
			ObjectInputStream oi = new ObjectInputStream(fi);

			// Read objects
			NeuralNetwork loadedNetwork = (NeuralNetwork) oi.readObject();

			oi.close();
			fi.close();

			return loadedNetwork;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error initializing stream");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	// load a neural network from a file that was saved as a plain text file
	// transferable between different programming languages
	public static NeuralNetwork loadParameters(String path) {
		try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
			String line;
			NeuralNetwork network = new NeuralNetwork();
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(" ");
				String paramType = tokens[0];

				switch (paramType) {
					case "numlayers":
						network.numLayers = Integer.parseInt(tokens[1]);
						network.neuronsPerLayer = new int[network.numLayers];
						network.activations = new String[network.numLayers];
						break;
					case "topology":
						int maxLayerSize = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.neuronsPerLayer[i - 1] = Integer.parseInt(tokens[i]);
							maxLayerSize = Math.max(maxLayerSize, network.neuronsPerLayer[i - 1]);
						}
						network.neurons = new double[network.numLayers][maxLayerSize];
						network.neuronsRaw = new double[network.numLayers][maxLayerSize];
						network.biases = new double[network.numLayers][maxLayerSize];
						network.weights = new double[network.numLayers][maxLayerSize][maxLayerSize];
						break;
					case "activations":
						for (int i = 1; i < tokens.length; i++) {
							network.activations[i - 1] = tokens[i];
						}
						break;
					case "regularization":
						network.regularizationType = RegularizationType.valueOf(tokens[1]);
						network.lambda = Double.parseDouble(tokens[2]);
						break;
					case "biases":
						int layerIndex = 0;
						int neuronIndex = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.biases[layerIndex][neuronIndex] = Double.parseDouble(tokens[i]);
							neuronIndex++;
							if (neuronIndex == network.neuronsPerLayer[layerIndex]) {
								neuronIndex = 0;
								layerIndex++;
							}
						}
						break;
					case "weights":
						layerIndex = 1;
						neuronIndex = 0;
						int incomingNeuronIndex = 0;
						for (int i = 1; i < tokens.length; i++) {
							network.weights[layerIndex][neuronIndex][incomingNeuronIndex] = Double
									.parseDouble(tokens[i]);
							incomingNeuronIndex++;
							if (incomingNeuronIndex == network.neuronsPerLayer[layerIndex - 1]) {
								incomingNeuronIndex = 0;
								neuronIndex++;
								if (neuronIndex == network.neuronsPerLayer[layerIndex]) {
									neuronIndex = 0;
									layerIndex++;
								}
							}
						}
						break;
				}
			}
			return network;
		} catch (FileNotFoundException e) {
			System.out.println("File not found");
		} catch (IOException e) {
			System.out.println("Error reading from file");
		} catch (ArrayIndexOutOfBoundsException e) {
			System.out.println("File not formatted correctly");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	// chance is a number between 0 and 1
	public void mutate(double chance, double variation) {
		// mutate weights
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				for (int k = 0; k < weights[0][0].length; k++) {
					if (randDouble(0, 1) <= chance) {
						weights[i][j][k] += randDouble(-variation, variation);
					}
				}
			}
		}
		// mutate biases
		for (int i = 0; i < biases.length; i++) {
			for (int j = 0; j < biases[0].length; j++) {
				if (randDouble(0, 1) <= chance) {
					biases[i][j] += randDouble(-variation, variation);
				}
			}
		}
	}

	public NeuralNetwork clone() {
		NeuralNetwork clone = new NeuralNetwork(neuronsPerLayer, activations, regularizationType, lambda);
		clone.biases = deepCopy(biases);
		clone.weights = deepCopy(weights);
		clone.clipThreshold = this.clipThreshold;
		clone.displayAccuracy = this.displayAccuracy;
		return clone;
	}

	// error functions
	public double cost(double[] output, double[] expected, String lossFunction) {
		double cost = 0;
		if (output.length != expected.length) {
			return -1;
		}
		if (lossFunction.equals("sse")) {
			for (int i = 0; i < output.length; i++) {
				double neuronCost = 0.5 * Math.pow(expected[i] - output[i], 2);
				cost += neuronCost;
			}
		} else if (lossFunction.equals("mse")) {
			for (int i = 0; i < output.length; i++) {
				double neuronCost = Math.pow(expected[i] - output[i], 2);
				cost += neuronCost;
			}
			cost /= output.length;
		} else if (lossFunction.equals("categorical_crossentropy")) {
			for (int i = 0; i < output.length; i++) {
				cost -= expected[i] * Math.log(output[i] + 1.0e-15d);
			}
		}
		// add regularization term
		cost += regularizationTerm();
		return cost;
	}

	// error functions derivative
	protected double cost_der(double predicted, double expected, String lossFunction) {
		if (lossFunction.equals("sse")) {
			return predicted - expected;
		} else if (lossFunction.equals("mse")) {
			return (2.0 * (predicted - expected)) / neuronsPerLayer[numLayers - 1];
		} else if (lossFunction.equals("categorical_crossentropy")) {
			return -expected / (predicted + 1.0e-15);
		}
		return 1;
	}

	private double[] backpropagate(double[][] biasGrad, double[][][] weightGrad, double[] predicted, double[] expected,
			String lossFunction) {
		return backpropagate(this.neurons, this.neuronsRaw, biasGrad, weightGrad, predicted, expected, 1, lossFunction);
	}

	private double[] backpropagate(double[][] neurons, double[][] neuronsRaw, double[][] biasGrad, double[][][] weightGrad,
			double[] predicted, double[] expected, int layer, String lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		// base case
		if (layer == numLayers - 1) {
			// last layer (output layer)
			for (int i = 0; i < neuronsPerLayer[layer]; i++) {
				if (lossFunction.equals("categorical_crossentropy") && activations[layer].equals("softmax")) {
					// Softmax with categorical crossentropy simplification to speed up computation
					neuronGradients[i] = predicted[i] - expected[i];
				} else {
					neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction)
							* activate_der(neuronsRaw[layer][i], layer, neuronsRaw[layer], i);
				}
				biasGrad[layer][i] = 1 * neuronGradients[i];
				for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
					weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
				}
			}
			return neuronGradients;
		}

		// recursive case
		double[] nextLayerBackpropagate = backpropagate(neurons, neuronsRaw, biasGrad, weightGrad, predicted, expected,
				layer + 1, lossFunction);
		double nextLayerSum = 0;
		double[] nextLayerWeightedSum = new double[neuronsPerLayer[layer]];
		for (int i = 0; i < neuronsPerLayer[layer + 1]; i++) {
			nextLayerSum += nextLayerBackpropagate[i];
		}
		for (int i = 0; i < neuronsPerLayer[layer]; i++) {
			for (int j = 0; j < neuronsPerLayer[layer + 1]; j++) {
				nextLayerWeightedSum[i] += nextLayerBackpropagate[j] * weights[layer + 1][j][i];
			}
			neuronGradients[i] = activate_der(neuronsRaw[layer][i], layer, i) * nextLayerWeightedSum[i];
			biasGrad[layer][i] = nextLayerSum;
			for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
				weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
			}
		}

		// return gradients of neurons in layer
		return neuronGradients;
	}

	public static double clamp(double value, double min, double max) {
		return Math.max(min, Math.min(max, value));
	}

	// for classification tasks, evaluates accuracy of the network on given dataset. outputs are expected to be one-hot encoded.
	public double evaluateAccuracy(double[][] inputs, double[][] outputs) {
		int numCorrect = 0;
		for (int i = 0; i < inputs.length; i++) {
			double[] predicted = evaluate(inputs[i]); // Predict the output for each input
			int prediction = indexOf(predicted, max(predicted));
			int actual = indexOf(outputs[i], max(outputs[i]));
			if (prediction == actual) {
				numCorrect++;
			}
		}
		return (double) numCorrect / inputs.length;
	}

	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			String lossFunction, double decay, Optimizer optimizer, TrainingCallback callback) {
		double lr = learningRate;
		// list of indices for data points, will be randomized in each epoch
		if (batchSize == -1) {
			batchSize = trainX.length;
		}
		List<Integer> indices = new ArrayList<Integer>(trainX.length);
		for (int i = 0; i < trainX.length; i++) {
			indices.add(i);
		}
		// initial shuffle
		Collections.shuffle(indices);
		// current index of data point
		int currentInd = 0;
		// precompute weighted average (multiply each element by this to average out all
		// data points in batch)
		final double weightedAvg = 1.0 / (double) batchSize;
		int epoch = 0;
		double progress = 0; // marks the epoch and progress through current epoch as decimal
		final int batchesPerEpoch = (int) Math.ceil((double) trainX.length / batchSize);
		int epochIteration = 0;
		if (trainX.length % batchSize != 0) {
			// batches wont divide evenly into samples
			System.out.println("warning: training data size is not divisible by sample size");
		}
		optimizer.initialize(neuronsPerLayer, biases, weights);
		avgBiasGradient = new double[numLayers][biases[0].length];
		avgWeightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
		double avgBatchTime = 0;
		int iteration = 0;
		// ThreadLocal variables for thread-specific arrays
		final ThreadLocal<double[][]> threadLocalNeurons = ThreadLocal
				.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalNeuronsRaw = ThreadLocal
				.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalBiasGradient = ThreadLocal
				.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][][]> threadLocalWeightGradient = ThreadLocal.withInitial(() -> {
			double[][][] gradients = new double[numLayers][][];
			for (int i = 0; i < numLayers; i++) {
				gradients[i] = new double[weights[i].length][weights[i][0].length];
			}
			return gradients;
		});
		// initialize accuracies
		double initialTrainAccuracy = 100 * evaluateAccuracy(trainX, trainY);
		double initialTestAccuracy = 100 * evaluateAccuracy(testX, testY);
		// round to 2 decimal
		initialTrainAccuracy = Math.round(initialTrainAccuracy * 100.0) / 100.0;
		initialTestAccuracy = Math.round(initialTestAccuracy * 100.0) / 100.0;
		if (callback != null) {
			callback.onEpochUpdate(0, 0, 0, initialTrainAccuracy, initialTestAccuracy);
		}
		AtomicInteger numCorrect = new AtomicInteger(0);
		// Initialize batchIndices once
		ArrayList<Integer> batchIndices = new ArrayList<>(batchSize);

		for (int i = 0; i < batchSize; i++) {
			batchIndices.add(0); // pre-fill with dummy values to avoid resizing
		}
		long startTime = System.currentTimeMillis();
		for (; epoch < epochs; iteration++) {
			// do epoch batch stuff (iteration is the current cumulative batch iteration)
			epochIteration = iteration % batchesPerEpoch;
			// do exponential learning rate decay
			lr = (1.0 / (1.0 + decay * iteration)) * learningRate;

			batchIndices.clear();
			// Use System.arraycopy for faster copying
			int endIndex = currentInd + batchSize;
			if (endIndex <= trainX.length) {
				// If the batch does not wrap around the end of the list
				batchIndices.addAll(indices.subList(currentInd, endIndex));
			} else {
				// If the batch wraps around the end of the list
				int wrapAroundIndex = endIndex % indices.size();
				batchIndices.addAll(indices.subList(currentInd, trainX.length));
				batchIndices.addAll(indices.subList(0, wrapAroundIndex));
			}

			numCorrect.set(0);
			if (iteration > 0) {
				// not first iteration, reset gradients
				for (int i = 1; i < numLayers; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						avgBiasGradient[i][j] = 0;
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							avgWeightGradient[i][j][k] = 0;
						}
					}
				}
			}

			// Parallelize this loop
			IntStream.range(0, batchSize).parallel().forEach(a -> {
				int caseInd = batchIndices.get(a);

				// Use thread-local arrays
				double[][] thisNeurons = threadLocalNeurons.get();
				double[][] thisNeuronsRaw = threadLocalNeuronsRaw.get();
				double[][] thisBiasGradient = threadLocalBiasGradient.get();
				double[][][] thisWeightGradient = threadLocalWeightGradient.get();

				// Calculate predicted output
				double[] predicted = evaluate(trainX[caseInd], thisNeurons, thisNeuronsRaw);

				// If this is a classification network, count the number correct
				if (displayAccuracy) {
					int prediction = indexOf(predicted, max(predicted));
					int actual = indexOf(trainY[caseInd], max(trainY[caseInd]));
					if (prediction == actual) {
						numCorrect.incrementAndGet();
					}
				}

				// Reset gradients
				for (int i = 1; i < numLayers; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						thisBiasGradient[i][j] = 0;
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							thisWeightGradient[i][j][k] = 0;
						}
					}
				}

				// Do backpropagation
				backpropagate(thisNeurons, thisNeuronsRaw, thisBiasGradient, thisWeightGradient, predicted,
						trainY[caseInd], 1, lossFunction);

				// Do weighted sum of gradients for average
				for (int i = 1; i < numLayers; i++) {
					for (int j = 0; j < neuronsPerLayer[i]; j++) {
						avgBiasGradient[i][j] += thisBiasGradient[i][j] * weightedAvg;
						for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
							avgWeightGradient[i][j][k] += thisWeightGradient[i][j][k] * weightedAvg;
						}
					}
				}
			});

			// gradient post-processing
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					// gradient clipping
					avgBiasGradient[i][j] = clamp(avgBiasGradient[i][j], -clipThreshold, clipThreshold);
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						// regularization
						if (regularizationType == RegularizationType.L1) {
							avgWeightGradient[i][j][k] += lambda * Math.signum(weights[i][j][k]);
						} else if (regularizationType == RegularizationType.L2) {
							avgWeightGradient[i][j][k] += lambda * weights[i][j][k];
						}
						// gradient clipping
						avgWeightGradient[i][j][k] = clamp(avgWeightGradient[i][j][k], -clipThreshold, clipThreshold);
					}
				}
			}
			optimizer.step(avgBiasGradient, avgWeightGradient, lr);
			currentInd += batchSize;
			boolean newEpoch = false;
			if (currentInd >= trainX.length) {
				// new epoch
				newEpoch = true;
				currentInd = 0;
				epoch++;
				Collections.shuffle(indices);
			}
			progress = epoch + currentInd / (double) trainX.length;
			if (displayAccuracy) {
				// calculate train accuracy
				double trainAccuracy = 100 * ((double) numCorrect.get() * weightedAvg);
				// round to 2 decimal
				trainAccuracy = Math.round(trainAccuracy * 100.0) / 100.0;
				// calculate test accuracy if new epoch
				double testAccuracy = -1;
				if (newEpoch) {
					testAccuracy = 100 * evaluateAccuracy(testX, testY);
					// round to 2 decimal
					testAccuracy = Math.round(testAccuracy * 100.0) / 100.0;
				}
				if (callback != null) {
					callback.onEpochUpdate(epoch + 1, epochIteration + 1, progress, trainAccuracy, testAccuracy);
				}
				progressBar(30, "Training", epoch + 1, epochs,
						(epochIteration + 1) + "/" + batchesPerEpoch + " accuracy: " + trainAccuracy + "%");
			} else {
				progressBar(30, "Training", epoch + 1, epochs, (epochIteration + 1) + "/" + batchesPerEpoch);
			}
		}
		long endTime = System.currentTimeMillis();
		avgBatchTime = (endTime - startTime) / (1000.0 * (iteration + 1));
		System.out.println();
		System.out.println("average batch time: " + avgBatchTime + " seconds");
	}

	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			String lossFunction, double decay, Optimizer optimizer) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay, optimizer, null);
	}

	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			String lossFunction, double decay) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay,
				new SGD());
	}

	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			String lossFunction) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, 0);
	}

	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			String lossFunction, Optimizer optimizer) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, 0, optimizer);
	}

	private void progressBar(int width, String title, int current, int total, String subtitle) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if (fill >= 0 && fill <= 1) {
			// set progress bar
			int fillAmount = (int) Math.ceil(fill * width);
			StringBuilder bar = new StringBuilder();
			bar.append(title).append(": ").append(filled.repeat(fillAmount)).append(unfilled.repeat(width - fillAmount))
					.append(" ").append(current).append("/").append(total).append(" ").append(subtitle).append("      ")
					.append("\r");
			System.out.print(bar.toString());
		}
	}

	private int max(int[] arr) {
		int m = -1;
		for (int i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	private double max(double[] arr) {
		double m = -1;
		for (double i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	private int indexOf(double[] arr, double v) {
		int index = -1;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}

	protected double[][] deepCopy(double[][] array) {
		double[][] copy = new double[array.length][];
		for (int i = 0; i < array.length; i++) {
			copy[i] = array[i].clone();
		}
		return copy;
	}

	protected double[][][] deepCopy(double[][][] array) {
		double[][][] copy = new double[array.length][][];
		for (int i = 0; i < array.length; i++) {
			copy[i] = new double[array[i].length][];
			for (int j = 0; j < array[i].length; j++) {
				copy[i][j] = array[i][j].clone();
			}
		}
		return copy;
	}
}