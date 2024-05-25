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
import java.util.Random;
import java.util.Collections;

import java.util.stream.IntStream;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;

	//first dimension is layer, second dimension is neuron # in layer
	protected double[][] neurons;
	protected double[][] neuronsRaw;
	protected double[][] biases;
	//first dimension is recieving layer, second dimension is recieving neuron #, third dimension is incoming neuron # from previous layer
	protected double[][][] weights;
	protected int[] neuronsPerLayer;
	/* activation choices:
		- linear
		- sigmoid
		- tanh
		- relu
		- binary
		- softmax
	*/
	protected String[] activations;
	public int numLayers;
	//gradient clipping threshold
	public double clipThreshold = 1;
	//whether or not to display accuracy while training (for classification models)
	public boolean displayAccuracy = false;

	// Regularization type
	public static enum RegularizationType {
		NONE,
		L1,
		L2
	}

	// Callback interface for training updates
	public static interface TrainingCallback {
		void onEpochUpdate(int epoch, int batch, double progress, double accuracy);
	}

	// Regularization settings (lambda = regularization strength)
	private double lambda = 0;
	private RegularizationType regularizationType = RegularizationType.NONE;
	//used in gradient descent
	volatile private double[][] avgBiasGradient;
	volatile private double[][][] avgWeightGradient;
	private Random r;

	//takes in int[] for number of neurons in each layer and string[] for activations of each layer
	public NeuralNetwork(int[] topology, String[] active) {
		int maxLayerSize = max(topology);
		neuronsPerLayer = topology.clone();
		numLayers = topology.length;
		neurons = new double[numLayers][maxLayerSize];
		neuronsRaw = new double[numLayers][maxLayerSize];
		biases = new double[numLayers][maxLayerSize];
		weights = new double[numLayers][maxLayerSize][maxLayerSize];
		activations = active.clone();
		r = new Random();
	}

	public NeuralNetwork(int[] topology, String[] active, RegularizationType regularizationType,
			double regularizationStrength) {
		this(topology, active);
		//set regularization
		this.regularizationType = regularizationType;
		lambda = regularizationStrength;
	}

	public NeuralNetwork() {

	}

	//initialize network with random starting values
	public void Init(double biasSpread) {
		ClearNeurons();
		InitWeights();
		InitBiases(biasSpread);
	}

	//initialize network with random starting values using a specified weight initialization method ('he' or 'xavier')
	public void Init(String weightInitMethod, double biasSpread) {
		ClearNeurons();
		InitWeights(weightInitMethod);
		InitBiases(biasSpread);
	}

	void InitWeights(String initMethod) {
		//initMethod is either "he" or "xavier"
		if (initMethod.equals("he")) {
			Random r = new Random();
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				//he weight initialization (for relu) (gaussian distribution)
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = r.nextGaussian() * std + mean;
					}
				}

			}
		} else if (initMethod.equals("xavier")) {
			for (int i = 1; i < numLayers; i++) {
				int n = neuronsPerLayer[i - 1];
				double min, max;
				//xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		} else {
			InitWeights();
		}
	}

	void InitWeights() {
		for (int i = 1; i < numLayers; i++) {
			int n = neuronsPerLayer[i - 1];
			double min, max;
			if (activations[i].equals("relu")) {
				//he weight initialization (for relu) (gaussian distribution)
				Random r = new Random();
				double mean = 0, std = Math.sqrt(2.0 / n);
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						weights[i][j][k] = r.nextGaussian() * std + mean;
					}
				}
			} else {
				//xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform distribution)
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

	void InitBiases(double spread) {
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				biases[i][j] = randDouble(-spread, spread);
			}
		}
	}

	public double[][][] GetWeights() {
		return weights;
	}

	public void SetWeight(int layer, int outgoing, int incoming, double value) {
		weights[layer][outgoing][incoming] = value;
	}

	public double[][] GetBiases() {
		return biases;
	}

	public void SetBias(int layer, int neuron, double bias) {
		biases[layer][neuron] = bias;
	}

	public String[] GetActivations() {
		return activations;
	}

	public void SetActivation(int layer, String act) {
		activations[layer] = act;
	}

	public double[][] GetNeurons() {
		return neurons;
	}

	public int[] GetTopology() {
		return neuronsPerLayer;
	}

	public void SetRegularizationType(RegularizationType regularizationType) {
		this.regularizationType = regularizationType;
	}

	public void SetRegularizationLambda(double lambda) {
		this.lambda = lambda;
	}

	public RegularizationType GetRegularizationType() {
		return regularizationType;
	}

	public double GetRegularizationLambda() {
		return lambda;
	}

	double randDouble(double min, double max) {
		return min + (max - min) * r.nextDouble();
	}

	int max(int[] arr) {
		int m = -1;
		for (int i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
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

	double linear_activation(double raw) {
		return raw;
	}

	double sigmoid_activation(double raw) {
		return 1d / (1 + Math.exp(-raw));
	}

	double tanh_activation(double raw) {
		return Math.tanh(raw);
	}

	double relu_activation(double raw) {
		return Math.max(0, raw);
	}

	double binary_activation(double raw) {
		return raw > 0 ? 1 : 0;
	}

	double softmax_activation(double raw, double[] neuronValues) {
		double maxVal = max(neuronValues);

		// Compute the normalization factor (sum of exponentials)
		double total = 0;
		for (double value : neuronValues) {
			total += Math.exp(value - maxVal);
		}
		// Compute the softmax activation
		return Math.exp(raw - maxVal - Math.log(total));
	}

	double softmax_der(double[] neuronValues, int index) {
		double softmax = softmax_activation(neuronValues[index], neuronValues);
		return softmax * (1.0 - softmax);
	}

	double activate(double raw, int layer) {
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

	double activate(double raw, int layer, double[] neuronsRaw) {
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

	double activate_der(double raw, int layer, int index) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				if (Double.isNaN(val)) {
					System.out.println("NaN error in sigmoid activation der");
				}
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				if (Double.isNaN(val)) {
					System.out.println("NaN error in tanh activation der");
				}
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
				if (Double.isNaN(val)) {
					System.out.println("NaN error in softmax activation der");
				}
				return val;
			default:
				return 1;
		}
	}

	double activate_der(double raw, int layer, double[] neuronsRaw, int index) {
		double val;
		switch (activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				double sigmoidVal = sigmoid_activation(raw);
				val = sigmoidVal * (1 - sigmoidVal);
				if (Double.isNaN(val)) {
					System.out.println("NaN error in sigmoid activation der");
				}
				return val;
			case "tanh":
				val = Math.pow(1d / Math.cosh(raw), 2);
				if (Double.isNaN(val)) {
					System.out.println("NaN error in tanh activation der");
				}
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
				if (Double.isNaN(val)) {
					System.out.println("NaN error in softmax activation der");
				}
				return val;
			default:
				return 1;
		}
	}

	void ClearNeurons() {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	void ClearNeurons(double[][] neurons, double[][] neuronsRaw) {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	public double[] Evaluate(double[] input, double[][] neurons, double[][] neuronsRaw) {
		ClearNeurons(neurons, neuronsRaw);

		// Set input neurons
        IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

        // Feed forward
        for (int layer = 1; layer < numLayers; layer++) {
            final int currentLayer = layer;  // Capture the current value of layer
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
                    neurons[currentLayer][i] = activate(neuronsRaw[currentLayer][i], currentLayer, Arrays.copyOfRange(neuronsRaw[currentLayer], 0, neuronsPerLayer[currentLayer]));
                });
            }
        }

		//return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	public double[] Evaluate(double[] input) {
		ClearNeurons();

		// Set input neurons
        IntStream.range(0, input.length).parallel().forEach(i -> neurons[0][i] = input[i]);

        // Feed forward
        for (int layer = 1; layer < numLayers; layer++) {
            final int currentLayer = layer;  // Capture the current value of layer
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

		//return output layer
		return Arrays.copyOfRange(neurons[numLayers - 1], 0, neuronsPerLayer[numLayers - 1]);
	}

	@Override
	public String toString() {
		String print = "Neural Network \n";
		print += "\nTopology (neurons per layer): " + printArr(neuronsPerLayer);
		print += "\nActivations (per layer): " + printArr(activations);
		print += "\nRegularization: " + regularizationType.toString() + " lambda: " + lambda;

		print += "\nBiases:\n";
		for (int i = 0; i < numLayers; i++) {
			print += "Layer " + (i + 1) + ": " + printArr(Arrays.copyOfRange(biases[i], 0, neuronsPerLayer[i])) + "\n";
		}

		print += "\nWeights:\n";
		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < neuronsPerLayer[i]; j++) {
				//each neuron
				print += "    Neuron " + (j + 1) + " of Layer " + (i + 1) + " Weights: \n"
						+ printArr(Arrays.copyOfRange(weights[i][j], 0, neuronsPerLayer[i - 1])) + "\n";
			}
		}
		return print;
	}

	String printArr(int[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		String print = "[";
		for (int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		return print;
	}

	String printArr(double[] arr) {
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

	String printArr(String[] arr) {
		if (arr == null)
			return "[]";
		if (arr.length == 0)
			return "[]";
		String print = "[";
		for (int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		return print;
	}

	public static void Save(NeuralNetwork network, String path) {
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
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static NeuralNetwork Load(String path) {
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
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	//chance is a number between 0 and 1
	public void Mutate(double chance, double variation) {
		//mutate weights
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				for (int k = 0; k < weights[0][0].length; k++) {
					if (randDouble(0, 1) <= chance) {
						weights[i][j][k] += randDouble(-variation, variation);
					}
				}
			}
		}
		//mutate biases
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
		clone.biases = biases.clone();
		clone.weights = weights.clone();
		return clone;
	}

	//error functions
	public double Cost(double[] output, double[] expected, String lossFunction) {
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
		//add regularization term
		cost += regularizationTerm();
		return cost;
	}

	//error functions derivative
	double cost_der(double predicted, double expected, String lossFunction) {
		if (lossFunction.equals("sse")) {
			if (Double.isNaN(predicted - expected)) {
				System.out.println("NaN error in cost derivative sse");
			}
			return predicted - expected;
		} else if (lossFunction.equals("mse")) {
			if (Double.isNaN((2.0 * (predicted - expected)) / neuronsPerLayer[numLayers - 1])) {
				System.out.println("NaN error in cost derivative mse");
			}
			return (2.0 * (predicted - expected)) / neuronsPerLayer[numLayers - 1];
		} else if (lossFunction.equals("categorical_crossentropy")) {
			if (Double.isNaN(-expected / (Math.max(predicted, 1.0e-15)))) {
				System.out.println("NaN error in cost derivative crossentropy: expected: " + expected + " predicted: "
						+ predicted);
			}
			return -expected / (predicted + 1.0e-15);
		}
		return 1;
	}

	double[] Backpropagate(double[][] biasGrad, double[][][] weightGrad, double[] predicted, double[] expected, String lossFunction) {
		return Backpropagate(this.neurons, this.neuronsRaw, biasGrad, weightGrad, predicted, expected, 1, lossFunction);
	}

	double[] Backpropagate(double[][] neurons, double[][] neuronsRaw, double[][] biasGrad, double[][][] weightGrad, double[] predicted, double[] expected, int layer, String lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		//base case
		if (layer == numLayers - 1) {
			//last layer (output layer)
			for (int i = 0; i < neuronsPerLayer[layer]; i++) {
				if (lossFunction.equals("categorical_crossentropy") && activations[layer].equals("softmax")) {
					// Softmax with categorical crossentropy simplification to speed up computation
					neuronGradients[i] = predicted[i] - expected[i];
				} else {
					neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction)
							* activate_der(neuronsRaw[layer][i], layer, neuronsRaw[layer], i);
				}
				if (Double.isNaN(neuronGradients[i])) {
					System.out.println("Nan error in neuron gradient of last layer. try reducing the learning rate");
					throw new ArithmeticException("NaN error");
				}
				biasGrad[layer][i] = 1 * neuronGradients[i];
				if (Double.isNaN(biasGrad[layer][i])) {
					System.out.println("Nan error in bias of last layer. try reducing the learning rate");
					throw new ArithmeticException("NaN error");
				}
				for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
					weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
					if (Double.isNaN(neurons[layer - 1][j])) {
						System.out.println(
								"Nan error in neuron value of second to last layer. try reducing the learning rate");
					}
					if (Double.isNaN(weightGrad[layer][i][j])) {
						System.out.println(
								"Nan error in weight of last layer. try reducing the learning rate: neuronGradient: "
										+ neuronGradients[i] + " neuron: " + neurons[layer - 1][j] + " cost der: "
										+ cost_der(predicted[i], expected[i], lossFunction) + " activate der: "
										+ activate_der(neuronsRaw[layer][i], layer, i) + " predicted: " + predicted[i]
										+ " expected: " + expected[i]);
						throw new ArithmeticException("NaN error");
					}
				}
			}
			return neuronGradients;
		}

		//recursive case
		double[] nextLayerBackpropagate = Backpropagate(neurons, neuronsRaw, biasGrad, weightGrad, predicted, expected, layer + 1, lossFunction);
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
			biasGrad[layer][i] = 1 * nextLayerSum;
			if (Double.isNaN(biasGrad[layer][i])) {
				System.out.println("Nan error in bias. try reducing the learning rate");
				throw new ArithmeticException("NaN error");
			}
			for (int j = 0; j < neuronsPerLayer[layer - 1]; j++) {
				weightGrad[layer][i][j] = neuronGradients[i] * neurons[layer - 1][j];
				if (Double.isNaN(weightGrad[layer][i][j])) {
					System.out.println("Nan error in weight. try reducing the learning rate");
					throw new ArithmeticException("NaN error");
				}
			}
		}

		//return gradients of neurons in layer
		return neuronGradients;
	}

	public static double clamp(double value, double min, double max) {
		return Math.max(min, Math.min(max, value));
	}

	//uses SGD (stochastic gradient descent)
	public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate, int batchSize,
			String lossFunction, double decay, double momentum, TrainingCallback callback) {
		double lr = learningRate;
		//list of indices for data points, will be randomized in each epoch
		List<Integer> indices = new ArrayList<Integer>(inputs.length);
		for (int i = 0; i < inputs.length; i++) {
			indices.add(i);
		}
		//initial shuffle
		Collections.shuffle(indices);
		//current index of data point
		int currentInd = 0;
		//precompute weighted average (multiply each element by this to average out all data points in batch)
		final double weightedAvg = 1.0 / (double) batchSize;
		int epoch = 0;
		double progress = 0; // marks the epoch and progress through current epoch as decimal
		final int batchesPerEpoch = (int) Math.ceil((double) inputs.length / batchSize);
		int epochIteration = 0;
		if (inputs.length % batchSize != 0) {
			//batches wont divide evenly into samples
			System.out.println("warning: training data size is not divisible by sample size");
		}
		//momentum vectors initialized to 0
		double[][] vBiases = new double[numLayers][biases[0].length];
		double[][][] vWeights = new double[numLayers][weights[0].length][weights[0][0].length];
		avgBiasGradient = new double[numLayers][biases[0].length];
		avgWeightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
		double avgBatchTime = 0;
		int iteration = 0;
		//ThreadLocal variables for thread-specific arrays
		final ThreadLocal<double[][]> threadLocalNeurons = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalNeuronsRaw = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][]> threadLocalBiasGradient = ThreadLocal.withInitial(() -> new double[numLayers][neurons[0].length]);
		final ThreadLocal<double[][][]> threadLocalWeightGradient = ThreadLocal.withInitial(() -> {
			double[][][] gradients = new double[numLayers][][];
			for (int i = 0; i < numLayers; i++) {
				gradients[i] = new double[weights[i].length][weights[i][0].length];
			}
			return gradients;
		});
		AtomicInteger numCorrect = new AtomicInteger(0);
		// Initialize batchIndices once
		ArrayList<Integer> batchIndices = new ArrayList<>(batchSize);

		for (int i = 0; i < batchSize; i++) {
			batchIndices.add(0); // pre-fill with dummy values to avoid resizing
		}
		for (; epoch < epochs; iteration++) {
			//do epoch batch stuff (iteration is the current cumulative batch iteration)
			epochIteration = iteration % batchesPerEpoch;

			if (batchSize == -1) {
				batchSize = inputs.length;
			}

			batchIndices.clear();
			// Use System.arraycopy for faster copying
			int endIndex = currentInd + batchSize;
			if (endIndex <= inputs.length) {
				// If the batch does not wrap around the end of the list
				batchIndices.addAll(indices.subList(currentInd, endIndex));
			} else {
				// If the batch wraps around the end of the list
				int wrapAroundIndex = endIndex % indices.size();
				batchIndices.addAll(indices.subList(currentInd, inputs.length));
				batchIndices.addAll(indices.subList(0, wrapAroundIndex));
			}

			// ArrayList<Integer> batchIndices = new ArrayList<Integer>(batchSize);
			// for (int i = 0; i < batchSize; i++) {
			// 	int index = (currentInd + i) % inputs.length;
			// 	batchIndices.add(indices.get(index));
			// }

			//int numCorrect = 0;
			numCorrect.set(0);
			if (iteration > 0) {
				//not first iteration, reset gradients
				for (double[] subarray : avgBiasGradient) {
					Arrays.fill(subarray, 0);
				}
				for (double[][] subarray : avgWeightGradient) {
					for (double[] subarray2 : subarray) {
						Arrays.fill(subarray2, 0);
					}
				}
			}
			double startTime = System.nanoTime();
			int[] count = new int[2];

			// Parallelize this loop
			IntStream.range(0, batchSize).parallel().forEach(a -> {
				int caseInd = batchIndices.get(a);

				// Use thread-local arrays
				double[][] thisNeurons = threadLocalNeurons.get();
				double[][] thisNeuronsRaw = threadLocalNeuronsRaw.get();
				double[][] thisBiasGradient = threadLocalBiasGradient.get();
				double[][][] thisWeightGradient = threadLocalWeightGradient.get();

				// Calculate predicted output
				double[] predicted = Evaluate(inputs[caseInd], thisNeurons, thisNeuronsRaw);

				// If this is a classification network, count the number correct
				if (displayAccuracy) {
					int prediction = indexOf(predicted, max(predicted));
					int actual = indexOf(outputs[caseInd], max(outputs[caseInd]));
					if (prediction == actual) {
						numCorrect.incrementAndGet();
					}
				}

				// Reset gradients
				for(int i = 0; i < numLayers; i++) {
					for(int j = 0; j < neurons.length; j++) {
						thisBiasGradient[i][j] = 0;
						for (int k = 0; k < neurons[0].length; k++) {
							thisWeightGradient[i][j][k] = 0;
						}
					}
				}

				// Do backpropagation
				Backpropagate(thisNeurons, thisNeuronsRaw, thisBiasGradient, thisWeightGradient, predicted,
						outputs[caseInd], 1, lossFunction);

				// Do weighted sum of gradients for average
				//synchronized (avgBiasGradient) {
				for (int i = 0; i < numLayers; i++) {
					for (int j = 0; j < biases[0].length; j++) {
						avgBiasGradient[i][j] += thisBiasGradient[i][j] * weightedAvg;
						for (int k = 0; k < weights[0][0].length; k++) {
							avgWeightGradient[i][j][k] += thisWeightGradient[i][j][k] * weightedAvg;
						}
					}
				}
				count[0]++;
				//}
			});
			System.out.println(count[0]);

			//use average gradients to find new parameters
			for (int i = 1; i < numLayers; i++) {
				for (int j = 0; j < neuronsPerLayer[i]; j++) {
					avgBiasGradient[i][j] = clamp(avgBiasGradient[i][j], -clipThreshold, clipThreshold);
					//do momentum
					vBiases[i][j] = momentum * vBiases[i][j] - avgBiasGradient[i][j] * lr;
					//apply velocity
					biases[i][j] = biases[i][j] + vBiases[i][j];
					for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
						avgWeightGradient[i][j][k] = clamp(avgWeightGradient[i][j][k], -clipThreshold, clipThreshold);
						// apply regularization gradient
						if (regularizationType == RegularizationType.L1) {
							avgWeightGradient[i][j][k] += lambda * Math.signum(weights[i][j][k]);
						} else if (regularizationType == RegularizationType.L2) {
							avgWeightGradient[i][j][k] += lambda * weights[i][j][k];
						}
						//do momentum
						vWeights[i][j][k] = momentum * vWeights[i][j][k] - avgWeightGradient[i][j][k] * lr;
						//apply velocity
						weights[i][j][k] = weights[i][j][k] + vWeights[i][j][k];
					}
				}
			}
			double endTime = System.nanoTime();
			double batchTime = (endTime - startTime) / 1e9;
			avgBatchTime += batchTime;
			currentInd += batchSize;
			if (currentInd >= inputs.length) {
				//new epoch
				currentInd = 0;
				epoch++;
				//do exponential learning rate decay
				lr = (1.0 / (1.0 + decay * epoch)) * learningRate;
				Collections.shuffle(indices);
			}
			progress = epoch + currentInd / (double) inputs.length;
			if (displayAccuracy) {
				double accuracy = 100 * ((double) numCorrect.get() * weightedAvg);
				//round to one decimal
				accuracy = Math.round(accuracy * 100.0) / 100.0;
				if (callback != null) {
					callback.onEpochUpdate(epoch + 1, epochIteration + 1, progress, accuracy);
				}
				progressBar(30, "Training", epoch + 1, epochs,
						(epochIteration + 1) + "/" + batchesPerEpoch + " accuracy: " + accuracy + "%");
			} else {
				progressBar(30, "Training", epoch + 1, epochs, (epochIteration + 1) + "/" + batchesPerEpoch);
			}
		}
		avgBatchTime /= (iteration + 1);
		System.out.println();
		System.out.println("Average batch time: " + avgBatchTime + " seconds");
	}

	void progressBar(int width, String title, int current, int total, String subtitle) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if (fill >= 0 && fill <= 1) {
			//set progress bar
			int fillAmount = (int) Math.ceil(fill * width);
			StringBuilder bar = new StringBuilder();
			bar.append(title).append(": ").append(filled.repeat(fillAmount)).append(unfilled.repeat(width - fillAmount)).append(" ").append(current).append("/").append(total).append(" ").append(subtitle).append(" ").append("\r");
			System.out.print(bar.toString());
		}
	}

	double max(double[] arr) {
		double m = -1;
		for (double i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	int indexOf(double[] arr, double v) {
		int index = -1;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}
}