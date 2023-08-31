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
	public int numLayers
	//gradient clipping threshold
	public double clipThreshold = 1;
	
	protected List<double[][]> biasGradient;
	protected List<double[][][]> weightGradient;

	//takes in int[] for number of neurons in each layer and string[] for activations of each layer
	public NeuralNetwork(int[] topology, String[] active) {
		int maxLayerSize = max(topology);
		neuronsPerLayer = topology.clone();
		numLayers = topology.length;
		neurons = new double[numLayers][maxLayerSize];
		neuronsRaw = new double[numLayers][maxLayerSize];
		biases = new double[numLayers][maxLayerSize];
		biasGradient = new ArrayList<double[][]>();
		weights = new double[numLayers][maxLayerSize][maxLayerSize];
		weightGradient = new ArrayList<double[][][]>();
		activations = active.clone();
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
    	double total = 0;
		for(double i : neuronValues) {
			total += Math.exp(i);
		}
		return Math.exp(raw) / Math.max(total, 1.0e-15);
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
			if(Double.isNaN(Math.exp(i))) {
				System.out.println("NaN error in softmax activation der: exp of neuron value is NaN");
			}
            total += Math.exp(i);
        }
		if(Double.isNaN(total)) {
			System.out.println("NaN error in softmax activation der: total exp value is NaN");
		}
        for (int i = 0; i < neuronValues.length; i++) {
			if(Double.isNaN(Math.exp(neuronValues[i]) / Math.max(total, 1.0e-15))) {
				System.out.println("NaN error in softmax activation der: softmax value is NaN");
				System.out.println("neuron value: " + neuronValues[i] + " total: " + total);
			}
			if(Double.isNaN(Math.max(total, 1.0e-15))) {
				System.out.println("NaN error in softmax activation der: max of total exp value is NaN");
			}
            softmaxValues[i] = Math.exp(neuronValues[i]) / Math.max(total, 1.0e-15);
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
				cost += 0.5 * Math.pow(expected[i] - output[i], 2);
			}
		} else if(lossFunction.equals("categorical_crossentropy")) {
			for(int i = 0; i < output.length; i++) {
				cost -= expected[i] * Math.log(Math.max(output[i], 1.0e-15d));
			}
		}
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

	double[] Backpropagate(double c, double[] predicted, double[] expected, int layer, int bGrad, int wGrad, String lossFunction) {
		double[] neuronGradients = new double[neuronsPerLayer[layer]];

		//base case
		if(layer == numLayers - 1) {
			//last layer (output layer)
			for(int i = 0; i < neuronsPerLayer[layer]; i++) {
				neuronGradients[i] = cost_der(predicted[i], expected[i], lossFunction) * activate_der(neuronsRaw[layer][i], layer, i);
			}
			//set weights/biases
			for(int i = 0; i < neuronsPerLayer[layer]; i++) {
				biasGradient.get(bGrad)[layer][i] = 1 * neuronGradients[i];
				if(Double.isNaN(biasGradient.get(bGrad)[layer][i])) {
					System.out.println("Nan error in bias of first layer. try reducing the learning rate");
					throw new ArithmeticException("NaN error");
				}
				for(int j = 0; j < neuronsPerLayer[layer-1]; j++) {
					weightGradient.get(wGrad)[layer][i][j] = neuronGradients[i] * neurons[layer-1][j];
					if(Double.isNaN(weightGradient.get(wGrad)[layer][i][j])) {
						System.out.println("Nan error in weight of first layer. try reducing the learning rate");
						throw new ArithmeticException("NaN error");
					}
				}	
			}
			return neuronGradients;
		}
		
		//recursive case
		double[] nextLayerBackpropagate = Backpropagate(c, predicted, expected, layer+1, bGrad, wGrad, lossFunction);
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
			biasGradient.get(bGrad)[layer][i] = 1 * nextLayerSum;
			if(Double.isNaN(biasGradient.get(bGrad)[layer][i])) {
				System.out.println("Nan error in bias. try reducing the learning rate");
				throw new ArithmeticException("NaN error");
			}
			for(int j = 0; j < neuronsPerLayer[layer-1]; j++) {
				weightGradient.get(wGrad)[layer][i][j] = neuronGradients[i] * neurons[layer-1][j];
				if(Double.isNaN(weightGradient.get(wGrad)[layer][i][j])) {
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
	public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate, int batchSize, String lossFunction, double decay) {
		Random r = new Random();
		double lr = learningRate;
		for(int e = 0; e < epochs; e++) {
			//do exponential learning rate decay
			lr = learningRate * Math.pow(1 - decay, e);
			
			biasGradient = new ArrayList<double[][]>();
			weightGradient = new ArrayList<double[][][]>();
			//choose random case

			if(batchSize == -1) {
				batchSize = inputs.length;
			}
			double[] c = new double[batchSize];
			int caseInd = 0;
			for(int i = 0; i < batchSize; i++) {
				caseInd = r.nextInt(inputs.length);
				
				double[] predicted = Evaluate(inputs[caseInd]);
				c[i] = Cost(predicted, outputs[caseInd], lossFunction);
			}
			double avgCost = 0;
			for(int i = 0; i < batchSize; i++) {
				avgCost += c[i];
			}
			avgCost /= (double)batchSize;
			for(int i = 0; i < batchSize; i++) {
				caseInd = r.nextInt(inputs.length);
				
				double[] predicted = Evaluate(inputs[caseInd]);
				biasGradient.add(new double[numLayers][neurons[0].length]);
				weightGradient.add(new double[numLayers][weights[0].length][weights[0][0].length]);
				Backpropagate(avgCost, predicted, outputs[caseInd], 1, biasGradient.size()-1, weightGradient.size()-1, lossFunction);
			}

			//average bias and weight gradients element wise
			double[][] avgBiasGradient = new double[numLayers][biases[0].length];
			double[][][] avgWeightGradient = new double[numLayers][weights[0].length][weights[0][0].length];
			for(int i = 0; i < numLayers; i++)  {
				for(int j = 0; j < biases[0].length; j++) {
					double avg = 0;
					for(double[][] bgrad : biasGradient) {
						avg += bgrad[i][j];
					}
					avg /= biasGradient.size();
					avgBiasGradient[i][j] = avg;
				}
			}
			for(int i = 0; i < numLayers; i++)  {
				for(int j = 0; j < weights[0].length; j++) {
					for(int k = 0; k < weights[0][0].length; k++) {
						double avg = 0;
						for(double[][][] wgrad : weightGradient) {
							avg += wgrad[i][j][k];
						}
						avg /= weightGradient.size();
						avgWeightGradient[i][j][k] = avg;
					}
				}
			}

			//use gradients to find new parameters
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
						weights[i][j][k] = weights[i][j][k] - avgWeightGradient[i][j][k] * lr;
					}
				}
			}
			progressBar(30, "Training", e+1, epochs);
		}
	}

	void progressBar(int width, String title, int current, int total) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if(fill >= 0 && fill <= 1) {
			//set progress bar
			int fillAmount = (int)Math.ceil(fill * width);
			String bar = title + ": " + filled.repeat(fillAmount) + unfilled.repeat(width - fillAmount) + " " + current + "/" + total + " " + "\r";
			System.out.print(bar);
		 }
	} 
}