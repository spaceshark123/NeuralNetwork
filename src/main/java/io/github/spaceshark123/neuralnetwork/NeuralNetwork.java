package io.github.spaceshark123.neuralnetwork;

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

import io.github.spaceshark123.neuralnetwork.activation.ActivationFunction;
import io.github.spaceshark123.neuralnetwork.activation.ReLU;
import io.github.spaceshark123.neuralnetwork.callback.TrainingCallback;
import io.github.spaceshark123.neuralnetwork.loss.LossFunction;
import io.github.spaceshark123.neuralnetwork.optimizer.Optimizer;
import io.github.spaceshark123.neuralnetwork.optimizer.SGD;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Represents a feedforward MLP (Multi-Layer Perceptron) neural network with customizable topology,
 * activation functions, weight initialization, and regularization.
 */
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
	protected ActivationFunction[] activations;
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

	public static enum WeightInitMethod {
		RANDOM,
		HE,
		XAVIER
	}

	// Regularization settings (lambda = regularization strength)
	protected double lambda = 0;
	protected RegularizationType regularizationType = RegularizationType.NONE;
	// used in gradient descent
	volatile protected double[][] avgBiasGradient;
	volatile protected double[][][] avgWeightGradient;

	/**
	 * Creates a neural network with the specified topology and activation
	 * functions.
	 * 
	 * @param topology array specifying the number of neurons in each layer
	 * @param active   array of activation functions for each layer
	 * @throws IllegalArgumentException if topology or activations are invalid
	 */
	public NeuralNetwork(int[] topology, ActivationFunction[] active) {
		if (topology == null) {
			throw new IllegalArgumentException("Topology cannot be null");
		}
		if (active == null) {
			throw new IllegalArgumentException("Activations cannot be null");
		}
		if (topology.length != active.length) {
			throw new IllegalArgumentException(
					"Topology and activations must have the same length (got " +
							topology.length + " and " + active.length + ")");
		}
		if (topology.length < 2) {
			throw new IllegalArgumentException(
					"Network must have at least 2 layers (input and output), got " + topology.length);
		}
		for (int i = 0; i < topology.length; i++) {
			if (topology[i] <= 0) {
				throw new IllegalArgumentException(
						"All layers must have at least 1 neuron. Layer " + i + " has " + topology[i]);
			}
		}

		int maxLayerSize = max(topology);
		neuronsPerLayer = topology.clone();
		numLayers = topology.length;
		neurons = new double[numLayers][maxLayerSize];
		neuronsRaw = new double[numLayers][maxLayerSize];
		biases = new double[numLayers][maxLayerSize];
		weights = new double[numLayers][maxLayerSize][maxLayerSize];
		activations = active.clone();
	}

	/**
	 * Creates a neural network with regularization.
	 * 
	 * @param topology               array specifying the number of neurons in each
	 *                               layer
	 * @param active                 array of activation functions for each layer
	 * @param regularizationType     type of regularization to use
	 * @param regularizationStrength strength of regularization (lambda)
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public NeuralNetwork(int[] topology, ActivationFunction[] active, RegularizationType regularizationType,
			double regularizationStrength) {
		this(topology, active);
		if (regularizationType == null) {
			throw new IllegalArgumentException("Regularization type cannot be null");
		}
		if (regularizationStrength < 0) {
			throw new IllegalArgumentException(
					"Regularization strength must be non-negative, got " + regularizationStrength);
		}
		// set regularization
		this.regularizationType = regularizationType;
		lambda = regularizationStrength;
	}

	public NeuralNetwork() {
		// Empty constructor for deserialization
	}

	public void init() {
		init(0);
	}

	/**
	 * Initializes network with random weights and biases.
	 * 
	 * @param biasSpread the range for bias initialization [-biasSpread, biasSpread]
	 */
	public void init(double biasSpread) {
		clearNeurons();
		initWeights();
		initBiases(biasSpread);
	}

	/**
	 * Initializes network with a specific weight initialization method.
	 * 
	 * @param weightInitMethod the weight initialization method
	 * @param biasSpread       the range for bias initialization
	 * @throws IllegalArgumentException if weightInitMethod is null
	 */
	public void init(WeightInitMethod weightInitMethod, double biasSpread) {
		clearNeurons();
		initWeights(weightInitMethod);
		initBiases(biasSpread);
	}

	protected void initWeights(WeightInitMethod initMethod) {
		// initMethod is either "he" or "xavier"
		if (initMethod == WeightInitMethod.HE) {
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
		} else if (initMethod == WeightInitMethod.XAVIER) {
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
		} else if (initMethod == WeightInitMethod.RANDOM) {
			for (int i = 1; i < numLayers; i++) {
				double min = -1;
				double max = 1;
				// random weight initialization (uniform distribution)
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		} else {
			initWeights(); // default initialization
		}
	}

	// default weight initialization (automatically chooses he or xavier based on
	// activation)
	protected void initWeights() {
		for (int i = 1; i < numLayers; i++) {
			int n = neuronsPerLayer[i - 1];
			double min, max;
			if (activations[i] instanceof ReLU) {
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

	/**
	 * Sets a specific weight value.
	 * 
	 * @param layer    the layer index of the outgoing neuron
	 * @param outgoing the outgoing neuron index
	 * @param incoming the incoming neuron index from previous layer
	 * @param value    the weight value
	 * @throws IllegalArgumentException if indices are out of bounds
	 */
	public void setWeight(int layer, int outgoing, int incoming, double value) {
		if (layer < 1 || layer >= numLayers) {
			throw new IllegalArgumentException(
					"Layer index must be between 1 and " + (numLayers - 1) + ", got " + layer);
		}
		if (outgoing < 0 || outgoing >= neuronsPerLayer[layer]) {
			throw new IllegalArgumentException(
					"Outgoing neuron index must be between 0 and " + (neuronsPerLayer[layer] - 1) +
							", got " + outgoing);
		}
		if (incoming < 0 || incoming >= neuronsPerLayer[layer - 1]) {
			throw new IllegalArgumentException(
					"Incoming neuron index must be between 0 and " + (neuronsPerLayer[layer - 1] - 1) +
							", got " + incoming);
		}
		if (!Double.isFinite(value)) {
			throw new IllegalArgumentException("Weight value must be finite, got " + value);
		}
		weights[layer][outgoing][incoming] = value;
	}

	public double[][] getBiases() {
		return deepCopy(biases);
	}

	/**
	 * Sets a specific bias value.
	 * 
	 * @param layer  the layer index
	 * @param neuron the neuron index
	 * @param bias   the bias value
	 * @throws IllegalArgumentException if indices are out of bounds
	 */
	public void setBias(int layer, int neuron, double bias) {
		if (layer < 1 || layer >= numLayers) {
			throw new IllegalArgumentException(
					"Layer index must be between 1 and " + (numLayers - 1) + ", got " + layer);
		}
		if (neuron < 0 || neuron >= neuronsPerLayer[layer]) {
			throw new IllegalArgumentException(
					"Neuron index must be between 0 and " + (neuronsPerLayer[layer] - 1) +
							", got " + neuron);
		}
		if (!Double.isFinite(bias)) {
			throw new IllegalArgumentException("Bias value must be finite, got " + bias);
		}
		biases[layer][neuron] = bias;
	}

	public ActivationFunction[] getActivations() {
		return activations.clone();
	}

	/**
	 * Sets the activation function for a specific layer.
	 * 
	 * @param layer the layer index
	 * @param act   the activation function
	 * @throws IllegalArgumentException if layer is out of bounds or activation is
	 *                                  null
	 */
	public void setActivation(int layer, ActivationFunction act) {
		if (layer < 0 || layer >= numLayers) {
			throw new IllegalArgumentException(
					"Layer index must be between 0 and " + (numLayers - 1) + ", got " + layer);
		}
		if (act == null) {
			throw new IllegalArgumentException("Activation function cannot be null");
		}
		activations[layer] = act;
	}

	public double[][] getNeurons() {
		return deepCopy(neurons);
	}

	public int[] getTopology() {
		return neuronsPerLayer.clone();
	}

	/**
	 * Sets the regularization type.
	 * 
	 * @param regularizationType the regularization type
	 * @throws IllegalArgumentException if regularizationType is null
	 */
	public void setRegularizationType(RegularizationType regularizationType) {
		if (regularizationType == null) {
			throw new IllegalArgumentException("Regularization type cannot be null");
		}
		this.regularizationType = regularizationType;
	}

	/**
	 * Sets the regularization strength (lambda).
	 * 
	 * @param lambda the regularization strength
	 * @throws IllegalArgumentException if lambda is negative
	 */
	public void setRegularizationLambda(double lambda) {
		if (lambda < 0) {
			throw new IllegalArgumentException(
					"Regularization lambda must be non-negative, got " + lambda);
		}
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
	protected double regularizationTerm() {
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

	protected double activate(double raw, int layer) {
		ActivationFunction activation = activations[layer];
		if (activation.requiresLayerContext()) {
			// pass full layer context but we dont have it, so throw error
			throw new UnsupportedOperationException(
					activation.getName() + " requires layer context. Use activateWithContext() instead.");
		}
		return activation.activate(raw);
	}

	protected double activate(double raw, int layer, double[] neuronsRaw, int index) {
		ActivationFunction activation = activations[layer];
		if (activation.requiresLayerContext()) {
			return activation.activateWithContext(raw, neuronsRaw, index);
		}
		return activation.activate(raw);
	}

	protected double activate_der(double raw, int layer, int index) {
		ActivationFunction activation = activations[layer];
		if (activation.requiresLayerContext()) {
			double[] layerRaw = Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]);
			return activation.derivativeWithContext(raw, layerRaw, index);
		}
		return activation.derivative(raw);
	}

	protected double activate_der(double raw, int layer, double[] neuronsRaw, int index) {
		ActivationFunction activation = activations[layer];
		if (activation.requiresLayerContext()) {
			return activation.derivativeWithContext(raw, neuronsRaw, index);
		}
		return activation.derivative(raw);
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

	/**
	 * Evaluates the network on the given input using provided neuron arrays to
	 * store
	 * intermediate neuron values. Useful for batch processing to avoid race
	 * conditions.
	 * 
	 * @param input      the input values
	 * @param neurons    the neuron array to use. Must match network topology.
	 * @param neuronsRaw the raw neuron array to use. Must match network topology.
	 * @return the output values
	 * @throws IllegalArgumentException if input is invalid
	 */
	public double[] evaluate(double[] input, double[][] neurons, double[][] neuronsRaw) {
		if (input == null) {
			throw new IllegalArgumentException("Input cannot be null");
		}
		if (input.length != neuronsPerLayer[0]) {
			throw new IllegalArgumentException(
					"Input size (" + input.length + ") does not match network input layer size (" +
							neuronsPerLayer[0] + ")");
		}
		for (int i = 0; i < input.length; i++) {
			if (!Double.isFinite(input[i])) {
				throw new IllegalArgumentException(
						"Input values must be finite. Value at index " + i + " is " + input[i]);
			}
		}

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
				// for context-requiring activations, store raw for now
				if (activations[currentLayer].requiresLayerContext()) {
					neurons[currentLayer][neuron] = raw;
				} else {
					neurons[currentLayer][neuron] = activate(raw, currentLayer);
				}
			});
			// handle context-requiring activations
			if (activations[currentLayer].requiresLayerContext()) {
				IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
					neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer,
							Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), i);
				});
			}
		}

		// return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	/**
	 * Evaluates the network on the given input.
	 * 
	 * @param input the input values
	 * @return the output values
	 * @throws IllegalArgumentException if input is invalid
	 */
	public double[] evaluate(double[] input) {
		if (input == null) {
			throw new IllegalArgumentException("Input cannot be null");
		}
		if (input.length != neuronsPerLayer[0]) {
			throw new IllegalArgumentException(
					"Input size (" + input.length + ") does not match network input layer size (" +
							neuronsPerLayer[0] + ")");
		}
		for (int i = 0; i < input.length; i++) {
			if (!Double.isFinite(input[i])) {
				throw new IllegalArgumentException(
						"Input values must be finite. Value at index " + i + " is " + input[i]);
			}
		}

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
				if (activations[currentLayer].requiresLayerContext()) {
					neurons[currentLayer][neuron] = raw;
				} else {
					neurons[currentLayer][neuron] = activate(raw, currentLayer);
				}
			});

			if (activations[currentLayer].requiresLayerContext()) {
				IntStream.range(0, neuronsPerLayer[currentLayer]).parallel().forEach(i -> {
					neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer,
							Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]), i);
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
		print.append("\nActivations (per layer): ")
				.append(printArr(Arrays.stream(activations).map(ActivationFunction::getName).toArray(String[]::new)));
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

	/**
	 * Saves the neural network to a file as a serialized Java object (non
	 * human-readable).
	 * 
	 * @param network the network to save
	 * @param path    the file path
	 * @throws IllegalArgumentException if network or path is null
	 */
	public static void save(NeuralNetwork network, String path) {
		if (network == null) {
			throw new IllegalArgumentException("Network cannot be null");
		}
		if (path == null || path.trim().isEmpty()) {
			throw new IllegalArgumentException("Path cannot be null or empty");
		}

		try {
			FileOutputStream f = new FileOutputStream(path);
			ObjectOutputStream o = new ObjectOutputStream(f);

			// Write objects to file
			o.writeObject(network);

			o.close();
			f.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + path);
			throw new RuntimeException("Failed to save network", e);
		} catch (IOException e) {
			System.out.println("Error saving network: " + e.getMessage());
			throw new RuntimeException("Failed to save network", e);
		}
	}

	/**
	 * Saves the neural network parameters to a human-readable text file.
	 * 
	 * @param network the network to save
	 * @param path    the file path
	 * @throws IllegalArgumentException if network or path is null
	 */
	public static void saveParameters(NeuralNetwork network, String path) {
		if (network == null) {
			throw new IllegalArgumentException("Network cannot be null");
		}
		if (path == null || path.trim().isEmpty()) {
			throw new IllegalArgumentException("Path cannot be null or empty");
		}

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
				print.append(network.activations[i].toConfigString()).append(" ");
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
		} catch (IOException e) {
			System.out.println("Error saving parameters: " + e.getMessage());
			throw new RuntimeException("Failed to save parameters", e);
		}
	}

	/**
	 * Loads a neural network from a serialized Java object file.
	 * 
	 * @param path the file path
	 * @return the loaded network
	 * @throws IllegalArgumentException if path is null
	 */
	public static NeuralNetwork load(String path) {
		if (path == null || path.trim().isEmpty()) {
			throw new IllegalArgumentException("Path cannot be null or empty");
		}

		try {
			FileInputStream fi = new FileInputStream(path);
			ObjectInputStream oi = new ObjectInputStream(fi);

			// Read objects
			NeuralNetwork loadedNetwork = (NeuralNetwork) oi.readObject();

			oi.close();
			fi.close();

			return loadedNetwork;
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + path);
			throw new RuntimeException("Failed to load network", e);
		} catch (IOException e) {
			System.out.println("Error loading network: " + e.getMessage());
			throw new RuntimeException("Failed to load network", e);
		} catch (ClassNotFoundException e) {
			System.out.println("Invalid network file format");
			throw new RuntimeException("Failed to load network", e);
		}
	}

	/**
	 * Loads a neural network from a parameter text file.
	 * 
	 * @param path the file path
	 * @return the loaded network
	 * @throws IllegalArgumentException if path is null
	 */
	public static NeuralNetwork loadParameters(String path) {
		if (path == null || path.trim().isEmpty()) {
			throw new IllegalArgumentException("Path cannot be null or empty");
		}

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
						network.activations = new ActivationFunction[network.numLayers];
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
							try {
								network.activations[i - 1] = ActivationFunction.fromConfigString(tokens[i]);
							} catch (IllegalArgumentException e) {
								throw new RuntimeException("Failed to load activation function: " + tokens[i], e);
							}
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
			System.out.println("File not found: " + path);
			throw new RuntimeException("Failed to load parameters", e);
		} catch (IOException e) {
			System.out.println("Error reading file: " + e.getMessage());
			throw new RuntimeException("Failed to load parameters", e);
		} catch (Exception e) {
			System.out.println("Error parsing file: " + e.getMessage());
			throw new RuntimeException("Failed to load parameters", e);
		}
	}

	/**
	 * Randomly mutates weights and biases with specified probability and magnitude.
	 * 
	 * @param chance    probability of mutation for each weight/bias (0 to 1)
	 * @param variation maximum magnitude of mutation
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public void mutate(double chance, double variation) {
		if (chance < 0 || chance > 1) {
			throw new IllegalArgumentException(
					"Chance must be between 0 and 1, got " + chance);
		}
		if (variation < 0) {
			throw new IllegalArgumentException(
					"Variation must be non-negative, got " + variation);
		}
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

	/**
	 * Computes the loss between output and expected values, including
	 * regularization term.
	 * 
	 * @param output       the network output
	 * @param expected     the expected output
	 * @param lossFunction the loss function to use
	 * @return the computed loss value
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public double loss(double[] output, double[] expected, LossFunction lossFunction) {
		if (output == null) {
			throw new IllegalArgumentException("Output cannot be null");
		}
		if (expected == null) {
			throw new IllegalArgumentException("Expected output cannot be null");
		}
		if (lossFunction == null) {
			throw new IllegalArgumentException("Loss function cannot be null");
		}
		if (output.length != expected.length) {
			throw new IllegalArgumentException(
					"Output and expected arrays must have the same length (got " +
							output.length + " and " + expected.length + ")");
		}

		double loss = lossFunction.compute(output, expected);
		// add regularization term
		loss += regularizationTerm();
		return loss;
	}

	// backpropagation algorithm to compute gradients
	private double[] backpropagate(double[][] neurons, double[][] neuronsRaw, double[][] biasGrad,
			double[][][] weightGrad,
			double[] predicted, double[] expected, int layer, LossFunction lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		// base case
		if (layer == numLayers - 1) {
			// last layer (output layer)
			double[] optimizedGradients = null;
			if (lossFunction.hasActivationOptimization()
					&& lossFunction.getOptimizedActivation().isInstance(activations[layer])) {
				optimizedGradients = lossFunction.optimizedGradient(predicted, expected);
			}
			if (optimizedGradients != null) {
				// use optimized gradient calculation
				for (int i = 0; i < neuronsPerLayer[layer]; i++) {
					neuronGradients[i] = optimizedGradients[i];
					biasGrad[layer][i] = neuronGradients[i];
					for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
						weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					}
				}
			} else {
				for (int i = 0; i < neuronsPerLayer[layer]; i++) {
					neuronGradients[i] = lossFunction.gradient(predicted, expected)[i]
							* activate_der(neuronsRaw[layer][i], layer, neuronsRaw[layer], i);
					biasGrad[layer][i] = neuronGradients[i];
					for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
						weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					}
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

	/**
	 * Evaluates classification accuracy on a dataset.
	 * 
	 * @param inputs  the input data
	 * @param outputs the expected outputs (one-hot encoded)
	 * @return the accuracy as a fraction between 0 and 1
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public double evaluateAccuracy(double[][] inputs, double[][] outputs) {
		if (inputs == null) {
			throw new IllegalArgumentException("Inputs cannot be null");
		}
		if (outputs == null) {
			throw new IllegalArgumentException("Outputs cannot be null");
		}
		if (inputs.length != outputs.length) {
			throw new IllegalArgumentException(
					"Inputs and outputs must have the same number of samples (got " +
							inputs.length + " and " + outputs.length + ")");
		}
		if (inputs.length == 0) {
			throw new IllegalArgumentException("Cannot evaluate accuracy on empty dataset");
		}

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

	/**
	 * Trains the neural network on the provided dataset.
	 * 
	 * @param trainX       training input data
	 * @param trainY       training output data
	 * @param testX        test input data
	 * @param testY        test output data
	 * @param epochs       number of training epochs
	 * @param learningRate initial learning rate
	 * @param batchSize    size of mini-batches (-1 for full batch)
	 * @param lossFunction loss function to use
	 * @param decay        learning rate decay factor
	 * @param optimizer    optimization algorithm to use
	 * @param callback     callback for training progress updates
	 * @throws IllegalArgumentException if parameters are invalid
	 */
	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			LossFunction lossFunction, double decay, Optimizer optimizer, TrainingCallback callback) {
		// Validate inputs
		if (trainX == null || trainY == null) {
			throw new IllegalArgumentException("Training data cannot be null");
		}
		if (testX == null || testY == null) {
			throw new IllegalArgumentException("Test data cannot be null");
		}
		if (trainX.length != trainY.length) {
			throw new IllegalArgumentException(
					"Training inputs and outputs must have same length (got " +
							trainX.length + " and " + trainY.length + ")");
		}
		if (testX.length != testY.length) {
			throw new IllegalArgumentException(
					"Test inputs and outputs must have same length (got " +
							testX.length + " and " + testY.length + ")");
		}
		if (trainX.length == 0) {
			throw new IllegalArgumentException("Training data cannot be empty");
		}
		if (epochs <= 0) {
			throw new IllegalArgumentException("Epochs must be positive, got " + epochs);
		}
		if (learningRate <= 0) {
			throw new IllegalArgumentException("Learning rate must be positive, got " + learningRate);
		}
		if (batchSize < -1 || batchSize == 0) {
			throw new IllegalArgumentException(
					"Batch size must be positive or -1 for full batch, got " + batchSize);
		}
		if (batchSize > trainX.length) {
			throw new IllegalArgumentException(
					"Batch size (" + batchSize + ") cannot exceed training data size (" + trainX.length + ")");
		}
		if (lossFunction == null) {
			throw new IllegalArgumentException("Loss function cannot be null");
		}
		if (decay < 0) {
			throw new IllegalArgumentException("Decay must be non-negative, got " + decay);
		}
		if (optimizer == null) {
			throw new IllegalArgumentException("Optimizer cannot be null");
		}

		// Validate data dimensions
		for (int i = 0; i < trainX.length; i++) {
			if (trainX[i] == null || trainX[i].length != neuronsPerLayer[0]) {
				throw new IllegalArgumentException(
						"Training input at index " + i + " has incorrect size");
			}
			if (trainY[i] == null || trainY[i].length != neuronsPerLayer[numLayers - 1]) {
				throw new IllegalArgumentException(
						"Training output at index " + i + " has incorrect size");
			}
		}
		for (int i = 0; i < testX.length; i++) {
			if (testX[i] == null || testX[i].length != neuronsPerLayer[0]) {
				throw new IllegalArgumentException(
						"Test input at index " + i + " has incorrect size");
			}
			if (testY[i] == null || testY[i].length != neuronsPerLayer[numLayers - 1]) {
				throw new IllegalArgumentException(
						"Test output at index " + i + " has incorrect size");
			}
		}

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

	// overloaded train method with no callback
	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			LossFunction lossFunction, double decay, Optimizer optimizer) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay, optimizer, null);
	}

	// overloaded train method with default optimizer (SGD) and no callback
	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			LossFunction lossFunction, double decay) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, decay,
				new SGD());
	}

	// overloaded train method with default optimizer (SGD), no decay, and no
	// callback
	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			LossFunction lossFunction) {
		train(trainX, trainY, testX, testY, epochs, learningRate, batchSize, lossFunction, 0);
	}

	// overloaded train method with no decay, and no callback
	public void train(double[][] trainX, double[][] trainY, double[][] testX, double[][] testY, int epochs,
			double learningRate, int batchSize,
			LossFunction lossFunction, Optimizer optimizer) {
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

	protected int max(int[] arr) {
		int m = -1;
		for (int i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	protected double max(double[] arr) {
		double m = -1;
		for (double i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	protected int indexOf(double[] arr, double v) {
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