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
	protected double[][] biasGradient;
	protected double[][][] weightGradient;

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
	}

	public NeuralNetwork(int[] topology, String[] active, RegularizationType regularizationType, double regularizationStrength) {
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

	void InitWeights() {
		for(int i = 1; i < numLayers; i++) {
			int n = neuronsPerLayer[i-1];
			double min, max;
			if(activations[i].equals("relu")) {
				//he weight initialization (for relu) (gaussian distribution)
				Random r = new Random();
				double mean = 0, std = Math.sqrt(2.0 / n);
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					for(int k = 0; k < neuronsPerLayer[i-1]; k++) {
						weights[i][j][k] = r.nextGaussian() * std + mean;
					}
				}
			} else {
				//xavier weight initialization (for linear, sigmoid, tanh, etc.) (uniform distribution)
				max = 1 / Math.sqrt(n);
				min = -max;
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					for(int k = 0; k < neuronsPerLayer[i-1]; k++) {
						weights[i][j][k] = randDouble(min, max);
					}
				}
			}
		}
	}

	void InitBiases(double spread) {
		for(int i = 1; i < numLayers; i++) {
			for(int j = 0; j < neuronsPerLayer[i]; j++) {
				biases[i][j] = randDouble(-spread, spread);
			}
		}
	}

	public double[][][] GetWeights() { return weights; }
	public void SetWeight(int layer, int outgoing, int incoming, double value) {
		weights[layer][outgoing][incoming] = value;
	}

	public double[][] GetBiases() { return biases; }
	public void SetBias(int layer, int neuron, double bias) { biases[layer][neuron] = bias; }

	public String[] GetActivations() { return activations; }
	public void SetActivation(int layer, String act) { activations[layer] = act; }

	public double[][] GetNeurons() { return neurons; }

	public int[] GetTopology() { return neuronsPerLayer; }

    public void SetRegularizationType(RegularizationType regularizationType) { this.regularizationType = regularizationType; }
	public void SetRegularizationLambda(double lambda) { this.lambda = lambda; }
	public RegularizationType GetRegularizationType() { return regularizationType; }
	public double GetRegularizationLambda() { return lambda; }

	double randDouble(double min, double max) {
		Random r = new Random();
		return min + (max - min) * r.nextDouble();
	}

	int max(int[] arr) {
		int m = -1;
		for(int i : arr) {
			if(i > m) {
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
			for(int i = 1; i < numLayers; i++) {
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					for(int k = 0; k < neuronsPerLayer[i-1]; k++) {
						regTerm += Math.abs(weights[i][j][k]);
					}
				}
			}
        } else if (regularizationType == RegularizationType.L2) {
            // L2 regularization
            for(int i = 1; i < numLayers; i++) {
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					for(int k = 0; k < neuronsPerLayer[i-1]; k++) {
						regTerm += weights[i][j][k]*weights[i][j][k];
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
		//this is a slightly modified version of softmax meant to normalize inputs by subtracting the max
		//this way, there are less nan overflow errors with the exponentiation
    	double maxVal = max(neuronValues);

		// Compute the normalization factor (sum of exponentials)
		double total = 0;
		for (double value : neuronValues) {
			total += Math.exp(value - maxVal);
		}
		// Compute the softmax activation
		return Math.exp(raw - maxVal) / Math.max(total, 1.0e-15);
	}

	double softmax_der(double[] neuronValues, int index) {
		for(int i = 0; i < neuronValues.length; i++) {
			if(Double.isNaN(neuronValues[i])) {
				System.out.println("NaN error in softmax activation der: input neuron value" + i + " is NaN"); 
			}
		}
		double[] softmaxValues = new double[neuronValues.length];
        double total = 0.0;

        // Calculate the softmax values
        for (double i : neuronValues) {
			if(Double.isNaN(Math.exp(i-max(neuronValues)))) {
				System.out.println("NaN error in softmax activation der: exp of neuron value is NaN");
			}
            total += Math.exp(i-max(neuronValues));
        }
		if(Double.isNaN(total)) {
			System.out.println("NaN error in softmax activation der: total exp value is NaN");
		}
        for (int i = 0; i < neuronValues.length; i++) {
			if(Double.isNaN(Math.exp(neuronValues[i]-max(neuronValues)) / Math.max(total, 1.0e-15))) {
				System.out.println("NaN error in softmax activation der: softmax value is NaN");
				System.out.println("neuron value: " + neuronValues[i] + " total: " + total);
			}
			if(Double.isNaN(Math.max(total, 1.0e-15))) {
				System.out.println("NaN error in softmax activation der: max of total exp value is NaN");
			}
            softmaxValues[i] = Math.exp(neuronValues[i]-max(neuronValues)) / Math.max(total, 1.0e-15);
        }

        // Calculate the derivative using the diagonal elements of the Jacobian matrix
        double[] softmaxDerivative = new double[neuronValues.length];
        for (int i = 0; i < neuronValues.length; i++) {
            softmaxDerivative[i] = softmaxValues[i] * (1.0 - softmaxValues[i]);
        }

        return softmaxDerivative[index];
	}

	double activate(double raw, int layer) {
		switch(activations[layer]) {
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
				return softmax_activation(raw, Arrays.copyOfRange(neurons[layer], 0, neuronsPerLayer[layer]));
			default:
				return linear_activation(raw);
		}
	}

	double activate_der(double raw, int layer, int index) {
		switch(activations[layer]) {
			case "linear":
				return 1;
			case "sigmoid":
				if(Double.isNaN(sigmoid_activation(raw) * (1 - sigmoid_activation(raw)))) {
					System.out.println("NaN error in sigmoid activation der");
				}
				return sigmoid_activation(raw) * (1 - sigmoid_activation(raw));
			case "tanh":
				if(Double.isNaN(Math.pow(1d / Math.cosh(raw), 2))) {
					System.out.println("NaN error in tanh activation der");
				}
				return Math.pow(1d / Math.cosh(raw), 2);
			case "relu":
				if(raw <= 0) {
					return 0;
				} else {
					return 1;
				}
			case "binary":
				return 0;
			case "softmax":
				if(Double.isNaN(softmax_der(Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]), index))) {
					System.out.println("NaN error in softmax activation der");
				}
				return softmax_der(Arrays.copyOfRange(neuronsRaw[layer], 0, neuronsPerLayer[layer]), index);
			default:
				return 1;
		}
	}

	void ClearNeurons() {
		for(int i = 0; i < numLayers; i++) {
			for(int j = 0; j < neurons[i].length; j++) {
				neurons[i][j] = 0;
				neuronsRaw[i][j] = 0;
			}
		}
	}

	public double[] Evaluate(double[] input) {
		ClearNeurons();
		//set input neurons
		for(int i = 0; i < input.length; i++) {
			neurons[0][i] = input[i];
		}

		//feed forwards
		for(int layer = 1; layer < numLayers; layer++) {
			for(int neuron = 0; neuron < neuronsPerLayer[layer]; neuron++) {
				double raw = biases[layer][neuron];
				for(int prevNeuron = 0; prevNeuron < neuronsPerLayer[layer-1]; prevNeuron++) {
					raw += weights[layer][neuron][prevNeuron] * neurons[layer-1][prevNeuron];
				}
				neuronsRaw[layer][neuron] = raw;
				if(activations[layer].equals("softmax")) {
					neurons[layer][neuron] = raw;
				} else {
					neurons[layer][neuron] = activate(raw, layer);
				}
			}
			if(activations[layer].equals("softmax")) {
				double[] activatedNeurons = new double[neuronsPerLayer[layer]];
				for(int i = 0; i < neuronsPerLayer[layer]; i++) {
					activatedNeurons[i] = activate(neurons[layer][i], layer);
				}
				for(int i = 0; i < activatedNeurons.length; i++) {
					neurons[layer][i] = activatedNeurons[i];
				}
			}
		}

		//return output layer
		return Arrays.copyOfRange(neurons[numLayers-1], 0, neuronsPerLayer[numLayers-1]);
	}

	@Override
	public String toString() {
		String print = "Neural Network \n";
		print += "\nTopology (neurons per layer): " + printArr(neuronsPerLayer);
		print += "\nActivations (per layer): " + printArr(activations);
		print += "\nRegularization: " + regularizationType.toString() + " lambda: " + lambda;
		
		print += "\nBiases:\n";
		for(int i = 0; i < numLayers; i++) {
			print += "Layer " + (i+1) + ": " + printArr(Arrays.copyOfRange(biases[i],0,neuronsPerLayer[i])) + "\n";
		}

		print += "\nWeights:\n";
		for(int i = 1; i < numLayers; i++) {
			for(int j = 0; j < neuronsPerLayer[i]; j++) {
				//each neuron
				print += "    Neuron " + (j+1) + " of Layer " + (i+1) + " Weights: \n" + printArr(Arrays.copyOfRange(weights[i][j],0,neuronsPerLayer[i-1])) + "\n"; 
			}
		}
		return print;
	}

	String printArr(int[] arr) {
		if(arr == null)
			return "[]";
		if(arr.length == 0)
			return "[]";
		String print = "[";
		for(int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		return print;
	}

	String printArr(double[] arr) {
		if(arr == null)
			return "[]";
		if(arr.length == 0)
			return "[]";
		String print = "[";
		for(int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		return print;
	}

	String printArr(String[] arr) {
		if(arr == null)
			return "[]";
		if(arr.length == 0)
			return "[]";
		String print = "[";
		for(int i = 0; i < arr.length - 1; i++) {
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
			network.biasGradient = null;
			network.weightGradient = null;
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
			loadedNetwork.biasGradient = null;
			loadedNetwork.weightGradient = null;
				
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
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[0].length; j++) {
				for(int k = 0; k < weights[0][0].length; k++) {
					if(randDouble(0,1) <= chance) {
						weights[i][j][k] += randDouble(-variation, variation);
					}
				}
			}
		}
		//mutate biases
		for(int i = 0; i < biases.length; i++) {
			for(int j = 0; j < biases[0].length; j++) {
				if(randDouble(0,1) <= chance) {
					biases[i][j] += randDouble(-variation, variation);
				}
			}
		}
	}

	//MSE (mean squared error) cost: 0.5(p[i] - e[i])^2
	public double Cost(double[] output, double[] expected, String lossFunction) {
		double cost = 0;
		if(output.length != expected.length) {
			return -1;
		}
		if(lossFunction.equals("mse")) {
			for(int i = 0; i < output.length; i++) {
				//System.out.println("neuron " + i + " output: " + output[i] + " expected: " + expected[i]);
				double neuronCost = 0.5 * Math.pow(expected[i] - output[i], 2);
				//System.out.println("neuron " + i + " + cost is " + neuronCost);
				cost += neuronCost;
			}
		} else if(lossFunction.equals("categorical_crossentropy")) {
			for(int i = 0; i < output.length; i++) {
				cost -= expected[i] * Math.log(Math.max(output[i], 1.0e-15d));
			}
		}
		//add regularization term
		cost += regularizationTerm();
		return cost;
	}

	//MSE derivative: (p[i] - e[i])
	double cost_der(double predicted, double expected, String lossFunction) {
		if(lossFunction.equals("mse")) {
			if(Double.isNaN(predicted - expected)) {
				System.out.println("NaN error in cost derivative mse");
			}
			return predicted - expected;
		} else if(lossFunction.equals("categorical_crossentropy")) {
			if(Double.isNaN(-expected / (Math.max(predicted, 1.0e-15)))) {
				System.out.println("NaN error in cost derivative crossentropy: expected: " + expected + " predicted: " + predicted);
			}
			return -expected / (Math.max(predicted, 1.0e-15));
		}
		return 1;
	}

	double[] Backpropagate(double c, double[] predicted, double[] expected, int layer, String lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		//base case
		if(layer == numLayers - 1) {
			//last layer (output layer)
			for(int i = 0; i < neuronsPerLayer[layer]; i++) {
				neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction) * activate_der(neuronsRaw[layer][i], layer, i);
			}
			//set weights/biases
			for(int i = 0; i < neuronsPerLayer[layer]; i++) {
				biasGradient[layer][i] = 1 * neuronGradients[i];
				if(Double.isNaN(biasGradient[layer][i])) {
					System.out.println("Nan error in bias of first layer. try reducing the learning rate");
					throw new ArithmeticException("NaN error");
				}
				for(int j = 0; j < neuronsPerLayer[layer-1]; j++) {
					weightGradient[layer][i][j] = neuronGradients[i] * neurons[layer-1][j];
					if(Double.isNaN(weightGradient[layer][i][j])) {
						System.out.println("Nan error in weight of first layer. try reducing the learning rate");
						throw new ArithmeticException("NaN error");
					}
				}	
			}
			return neuronGradients;
		}
		
		//recursive case
		double[] nextLayerBackpropagate = Backpropagate(c, predicted, expected, layer+1, lossFunction);
		double nextLayerSum = 0;
		double[] nextLayerWeightedSum = new double[neuronsPerLayer[layer]];
		for(int i = 0; i < neuronsPerLayer[layer+1]; i++) {
			nextLayerSum += nextLayerBackpropagate[i];
		}
		for(int i = 0; i < neuronsPerLayer[layer]; i++) {
			for(int j = 0; j < neuronsPerLayer[layer+1]; j++) {
				nextLayerWeightedSum[i] += nextLayerBackpropagate[j] * weights[layer+1][j][i];
			}
		}
		//update gradients
		for(int i = 0; i < neuronsPerLayer[layer]; i++) {
			neuronGradients[i] = activate_der(neuronsRaw[layer][i], layer, i) * nextLayerWeightedSum[i];
			biasGradient[layer][i] = 1 * nextLayerSum;
			if(Double.isNaN(biasGradient[layer][i])) {
				System.out.println("Nan error in bias. try reducing the learning rate");
				throw new ArithmeticException("NaN error");
			}
			for(int j = 0; j < neuronsPerLayer[layer-1]; j++) {
				weightGradient[layer][i][j] = neuronGradients[i] * neurons[layer-1][j];
				if(Double.isNaN(weightGradient[layer][i][j])) {
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
	public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate, int batchSize, String lossFunction, double decay, TrainingCallback callback) {
		double lr = learningRate;
		//list of indices for data points, will be randomized in each epoch
		List<Integer> indices = new ArrayList<Integer>(inputs.length);
		for(int i = 0; i < inputs.length; i++) {
			indices.add(i);
		}
		//initial shuffle
		Collections.shuffle(indices);
		//current index of data point
		int currentInd = 0;
		//precompute weighted average (multiply each element by this to average out all data points in batch)
		final double weightedAvg = 1.0 / (double) batchSize;
		//System.out.println("clip threshold: " + clipThreshold);
		int epoch = 0;
		double progress = 0; // marks the epoch and progress through current epoch as decimal
		int batchesPerEpoch = (int)Math.ceil((double)inputs.length / batchSize);
		int epochIteration = 0;
		if(inputs.length % batchSize != 0) {
			//batches wont divide evenly into samples
			System.out.println("warning: training data size is not divisible by sample size");
		}
		for(int iteration = 0; epoch < epochs; iteration++) {
			//do epoch batch stuff (iteration is the current cumulative batch iteration)
			progress = (double)iteration*batchSize / inputs.length;
			epochIteration = iteration % batchesPerEpoch;
			epoch = (int)Math.floor(progress);

			//do exponential learning rate decay
			lr = (1.0/(1.0+decay*epoch))*learningRate;
			
			//choose random case
			if(batchSize == -1) {
				batchSize = inputs.length;
			}
			double numCorrect = 0;
			int caseInd = 0;
			//System.out.println("weighted avg " + weightedAvg);
			double[][] avgBiasGradient = new double[numLayers][biases[0].length];
			double[][][] avgWeightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
			for(int a = 0; a < batchSize; a++) {
				//find current random data point
				if(currentInd >= indices.size()) {
					currentInd = 0;
					//finished epoch, start new epoch by reshuffling
					Collections.shuffle(indices);
				}
				caseInd = indices.get(currentInd);
				currentInd++;
				
				//calculate predicted output
				double[] predicted = Evaluate(inputs[caseInd]);

				//if this is a classification network, count the number correct
				if(displayAccuracy) {
					int prediction = indexOf(predicted, max(predicted));
					int actual = indexOf(outputs[caseInd], max(outputs[caseInd]));
					if(prediction == actual) {
						numCorrect++;
					}
				}

				biasGradient = new double[numLayers][neurons[0].length];
				weightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
				//do backpropagation
				Backpropagate(Cost(predicted, outputs[caseInd], lossFunction), predicted, outputs[caseInd], 1, lossFunction);
				//do weighted sum of gradients for average
				for(int i = 0; i < numLayers; i++)  {
					for(int j = 0; j < biases[0].length; j++) {
						avgBiasGradient[i][j] += biasGradient[i][j] * weightedAvg;
					}
				}
				for(int i = 0; i < numLayers; i++)  {
					for(int j = 0; j < weights[0].length; j++) {
						for(int k = 0; k < weights[0][0].length; k++) {
							avgWeightGradient[i][j][k] += weightGradient[i][j][k] * weightedAvg;
						}
					}
				}
			}

			//use average gradients to find new parameters
			for(int i = 0; i < numLayers; i++) {
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					avgBiasGradient[i][j] = clamp(avgBiasGradient[i][j], -clipThreshold, clipThreshold);
					biases[i][j] = biases[i][j] - avgBiasGradient[i][j] * lr;
				}
			}
			for(int i = 1; i < numLayers; i++) {
				for(int j = 0; j < neuronsPerLayer[i]; j++) {
					for(int k = 0; k < neuronsPerLayer[i-1]; k++) {
						avgWeightGradient[i][j][k] = clamp(avgWeightGradient[i][j][k], -clipThreshold, clipThreshold);
						// apply regularization gradient
						if (regularizationType == RegularizationType.L1) {
							avgWeightGradient[i][j][k] += lambda * Math.signum(weights[i][j][k]);
						} else if (regularizationType == RegularizationType.L2) {
							avgWeightGradient[i][j][k] += lambda * weights[i][j][k];
						}
						weights[i][j][k] = weights[i][j][k] - avgWeightGradient[i][j][k] * lr;
					}
				}
			}
			if(displayAccuracy) {
				double accuracy = 100*((double) numCorrect * weightedAvg);
				//round to one decimal
				accuracy = Math.round(accuracy * 100.0) / 100.0;
				if (callback != null) {
					callback.onEpochUpdate(epoch + 1, epochIteration + 1, progress, accuracy);
				}
				progressBar(30, "Training", epoch+1, epochs, (epochIteration+1) + "/" + batchesPerEpoch + " accuracy: "+accuracy+"%");
			} else {
				progressBar(30, "Training", epoch+1, epochs, (epochIteration+1) + "/" + batchesPerEpoch);
			}
		}
	}

	void progressBar(int width, String title, int current, int total, String subtitle) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if(fill >= 0 && fill <= 1) {
			//set progress bar
			int fillAmount = (int)Math.ceil(fill * width);
			String bar = title + ": " + filled.repeat(fillAmount) + unfilled.repeat(width - fillAmount) + " " + current + "/" + total + " " + subtitle + " " + "\r";
			System.out.print(bar);
		 }
	} 

	double max(double[] arr) {
		double m = -1;
		for(double i : arr) {
			if(i > m) {
				m = i;
			}
		}
		return m;
	}

	int indexOf(double[] arr, double v) {
		int index = -1;
		for(int i = 0; i < arr.length; i++) {
			if(arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}
}