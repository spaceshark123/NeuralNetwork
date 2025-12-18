package com.github.spaceshark123.neuralnetwork.cli;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.GridLayout;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.StringTokenizer;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import com.github.spaceshark123.neuralnetwork.NeuralNetwork;
import com.github.spaceshark123.neuralnetwork.activation.ActivationFunction;
import com.github.spaceshark123.neuralnetwork.callback.ChartUpdater;
import com.github.spaceshark123.neuralnetwork.loss.*;
import com.github.spaceshark123.neuralnetwork.optimizer.*;
import com.github.spaceshark123.neuralnetwork.util.RealTimeSoftDrawCanvas;

class NeuralNetworkCLI {
	public static int SIZE = 1000;
	public static double[][] mnistOutputs;
	public static int[] mnistLabels;
	public static double[][] mnistImages;
	public static boolean mnistInitialized = false;

	public static void main(String[] args) {
		// load?
		Scanner scan = new Scanner(System.in);
		NeuralNetwork nn = new NeuralNetwork();
		Clear();

		while (true) {
			String[] s = Input(scan).split(" ");
			try {
				if (s[0].equals("load")) {
					if (s.length > 1) {
						if (s.length > 2) {
							if (s[2].equals("object")) {
								// object mode
								Output("loading...");
								nn = NeuralNetwork.load(s[1]);
								continue;
							} else if (s[2].equals("parameters")) {
								// parameter mode
								Output("loading...");
								nn = NeuralNetwork.loadParameters(s[1]);
								continue;
							} else {
								// object or parameter mode not specified
								Output("please specify a valid mode: 'object' or 'parameters'");
								continue;
							}
						} else {
							// object or parameter mode not specified
							Output("please specify a mode: 'object' or 'parameters'");
						}
					} else {
						Output("please specify a path and mode: 'object' or 'parameters'");
					}
				} else if (s[0].equals("save")) {
					if (s.length > 1) {
						if (s.length > 2) {
							if (s[2].equals("object")) {
								// object mode
								Output("saving...");
								NeuralNetwork.save(nn, s[1]);
								continue;
							} else if (s[2].equals("parameters")) {
								// parameter mode
								Output("saving...");
								NeuralNetwork.saveParameters(nn, s[1]);
								continue;
							} else {
								// object or parameter mode not specified
								Output("please specify a valid mode: 'object' or 'parameters'");
								continue;
							}
						} else {
							// object or parameter mode not specified
							Output("please specify a mode: 'object' or 'parameters'");
						}
					} else {
						Output("please specify a path and mode: 'object' or 'parameters'");
					}
				} else if (s[0].equals("mnist") && s[1].equals("test")) {
					// command mnist test was called
					testMnist(nn);
				} else if (s[0].equals("exit")) {
					Output("exiting...");
					break;
				} else if (s[0].equals("info")) {
					// print info about neural network
					if (s.length == 1) {
						System.out.println(nn);
					} else {
						if (s[1].equals("activations")) {
							System.out.print("activations: ");
							printArr(Arrays.stream(nn.getActivations()).map(ActivationFunction::getName)
									.toArray(String[]::new));
						} else if (s[1].equals("topology")) {
							System.out.print("topology: ");
							printArr(nn.getTopology());
						} else if (s[1].equals("regularization")) {
							System.out.println("regularization: " + nn.getRegularizationType().toString() + " lambda: "
									+ nn.getRegularizationLambda());
						} else if (s[1].equals("biases")) {
							String print = "";
							print += "\nBiases:\n";
							for (int i = 0; i < nn.numLayers; i++) {
								print += "Layer " + (i + 1) + ": "
										+ returnArr(Arrays.copyOfRange(nn.getBiases()[i], 0, nn.getTopology()[i]))
										+ "\n";
							}
							Output(print);
						} else if (s[1].equals("weights")) {
							String print = "";
							print += "\nWeights:\n";
							for (int i = 1; i < nn.numLayers; i++) {
								for (int j = 0; j < nn.getTopology()[i]; j++) {
									// each neuron
									print += "    Neuron "
											+ (j + 1) + " of Layer " + (i + 1) + " Weights: \n" + returnArr(Arrays
													.copyOfRange(nn.getWeights()[i][j], 0, nn.getTopology()[i - 1]))
											+ "\n";
								}
							}
							Output(print);
						} else {
							Output("not a valid property. choices are:\n    - activations\n    - topology\n    - biases\n    - weights");
						}
					}
				} else if (s[0].equals("clear")) {
					// clear console
					Clear();
					Output("cleared console");
				} else if (s[0].equals("create")) {
					// create a new neural network
					Output("Number of layers (at least 2): ");
					int[] topology = new int[Integer.parseInt(Input(scan))];
					Output("Size of each layer (separated by spaces): ");
					String[] ls = Input(scan).split(" ");
					if (ls.length != topology.length) {
						Output("mismatch in number of layers");
						continue;
					}
					for (int i = 0; i < ls.length; i++) {
						topology[i] = Integer.parseInt(ls[i]);
					}
					Output("Activations for each layer (separated by spaces): ");
					Output("Choices: \n    - linear\n    - sigmoid\n    - tanh\n    - relu\n    - leakyrelu(alpha={value})\n    - binary\n    - softmax");
					ls = Input(scan).split(" ");
					if (ls.length != topology.length) {
						Output("mismatch in number of layers");
						continue;
					}
					ActivationFunction[] as = new ActivationFunction[topology.length];
					for (int i = 0; i < ls.length; i++) {
						try {
							as[i] = ActivationFunction.fromConfigString(ls[i].toUpperCase());
						} catch (IllegalArgumentException e) {
							Output("invalid activation function: " + ls[i]);
							continue;
						}
					}
					nn = new NeuralNetwork(topology, as);
					Output("Created neural network");
				} else if (s[0].equals("init")) {
					// initialize neural network
					if (s.length >= 2) {
						Output("initializing weights and biases...");
						nn.init(Double.parseDouble(s[1]));
						if (s.length >= 3) {
							nn.init(NeuralNetwork.WeightInitMethod.valueOf(s[2].toUpperCase()),
									Double.parseDouble(s[1]));
						}
					} else {
						Output("please specify a bias spread and optionally a weight initialization method ('he' or 'xavier')");
						continue;
					}
				} else if (s[0].equals("regularization")) {
					// modify regularization
					if (s.length < 2) {
						// no argument is provided
						Output("please specify a valid property to modify: \n    - type\n    - lambda");
						continue;
					}
					if (s[1].equals("type")) {
						if (s.length < 3) {
							// no type is provided
							Output("please specify a regularization to apply (not case sensitive): \n    - none\n    - L1\n    - L2");
							continue;
						}
						if (s[2].equalsIgnoreCase("none")) {
							nn.setRegularizationType(NeuralNetwork.RegularizationType.NONE);
							Output("applying regularization...");
						} else if (s[2].equalsIgnoreCase("L1")) {
							nn.setRegularizationType(NeuralNetwork.RegularizationType.L1);
							Output("applying regularization...");
						} else if (s[2].equalsIgnoreCase("L2")) {
							nn.setRegularizationType(NeuralNetwork.RegularizationType.L2);
							Output("applying regularization...");
						} else {
							// invalid type is provided
							Output("please specify a valid regularization to apply (not case sensitive): \n    - none\n    - L1\n    - L2");
							continue;
						}
					} else if (s[1].equals("lambda")) {
						if (s.length < 3) {
							// no type is provided
							Output("please specify a regularization lambda (strength) to set.");
							continue;
						}
						nn.setRegularizationLambda(Double.parseDouble(s[2]));
						Output("setting lambda...");
					} else {
						// invalid argument is provided
						Output("please specify a valid property to modify: \n    - type\n    - lambda");
						continue;
					}
				} else if (s[0].equals("evaluate")) {
					if (s.length > 2) {
						if (s[1].equals("mnist")) {
							if (!mnistInitialized) {
								Output("the mnist dataset has not yet been initialized. run 'mnist'");
								continue;
							}
							int index = Integer.parseInt(s[2]);
							showImage(mnistImages[index], 28, 28);
							double[] output = nn.evaluate(mnistImages[index]);
							Output("predicted: " + indexOf(output, max(output)));
							Output("actual: " + mnistLabels[index]);
							System.out.print("output: ");
							printArr(output);
							continue;
						}
					}
					Output(nn.getTopology()[0] + " input(s) (separated by spaces): ");
					String[] ls = Input(scan).split(" ");
					double[] input = new double[nn.getTopology()[0]];
					if (ls.length != nn.getTopology()[0]) {
						Output("mismatch in number of inputs");
						continue;
					}
					for (int i = 0; i < ls.length; i++) {
						input[i] = Double.parseDouble(ls[i]);
					}
					printArr(nn.evaluate(input));
				} else if (s[0].equals("reset")) {
					Output("resetting network...");
					nn = new NeuralNetwork();
				} else if (s[0].equals("modify")) {
					if (s.length > 1) {
						if (s[1].equals("activations")) {
							Output("enter the layer to modify (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if (layer >= nn.numLayers || layer < 0) {
								// not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the new activation: ");
							Output("Choices: \n    - linear\n    - sigmoid\n    - tanh\n    - relu\n    - binary\n    - softmax");
							try {
								nn.setActivation(layer, ActivationFunction.fromConfigString(Input(scan).toUpperCase()));
							} catch (IllegalArgumentException e) {
								Output("invalid activation function");
								continue;
							}
							Output("making modification...");
						} else if (s[1].equals("weights")) {
							Output("enter the layer of the end neuron (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if (layer >= nn.numLayers || layer < 0) {
								// not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the neuron # of the end neuron (1 is first neuron): ");
							int end = Integer.parseInt(Input(scan)) - 1;
							if (end >= nn.getTopology()[layer] || end < 0) {
								// not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the neuron # of the start neuron from the previous layer (1 is first neuron): ");
							int start = Integer.parseInt(Input(scan)) - 1;
							if (start >= nn.getTopology()[layer - 1] || start < 0) {
								// not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the new weight: ");
							nn.setWeight(layer, end, start, Double.parseDouble(Input(scan)));
							Output("setting weight...");
						} else if (s[1].equals("biases")) {
							Output("enter the layer of the neuron (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if (layer >= nn.numLayers || layer < 0) {
								// not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the neuron # (1 is first neuron): ");
							int end = Integer.parseInt(Input(scan)) - 1;
							if (end >= nn.getTopology()[layer] || end < 0) {
								// not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the new bias: ");
							nn.setBias(layer, end, Double.parseDouble(Input(scan)));
							Output("setting bias...");
						} else {
							// modify argument isnt valid
							Output("please specify a valid property to modify: \n    - activations\n    - weights\n    - biases");
							continue;
						}
					} else {
						// no modify argument is provided
						Output("please specify a valid property to modify: \n    - activations\n    - weights\n    - biases");
						continue;
					}
				} else if (s[0].equals("mutate")) {
					if (s.length >= 3) {
						Output("mutating...");
						nn.mutate(Double.parseDouble(s[1]), Double.parseDouble(s[2]));
					} else {
						Output("please specify the mutation chance (decimal) and variation");
						continue;
					}
				} else if (s[0].equals("train")) {
					if (s.length < 7) {
						Output("please specify the following:\n    - path to the training set or 'mnist'\n    - train/test split (ratio between 0 and 1)\n    - number of epochs\n    - learning rate\n    - loss function\n    - optimizer\n    - 'custom' (optional to specify custom optimizer hyperparameters)\n    - batch size (optional)\n    - decay rate (optional)\n    - clip threshold (optional)");
						continue;
					}
					// trainset file must be formatted:
					// first line has 3 numbers specifying # cases, input and output size
					// every line is a separate training case
					// on each line, input is separated by spaces, then equal sign, then output
					// separated by spaces
					try {
						if (s[1].equals("mnist")) {
							// train on mnist
							if (!mnistInitialized) {
								Output("the mnist dataset has not yet been initialized. run 'mnist'");
								continue;
							} else {
								if (!(s[6].equals("sgd") || s[6].equals("sgdmomentum") || s[6].equals("adagrad")
										|| s[6].equals("rmsprop") || s[6].equals("adam"))) {
									// invalid loss function
									Output("invalid optimizer. choices are:\n    - sgd\n    - sgdmomentum\n    - adagrad\n    - rmsprop\n    - adam");
									continue;
								}
								int ind = 7;
								Optimizer optimizer;
								switch (s[6]) {
									case "sgd":
										optimizer = new SGD();
										break;
									case "sgdmomentum":
										// check if custom arguments are provided
										double momentum = 0.9;
										if (s.length >= 8) {
											if (s[7].equals("custom")) {
												// use custom momentum
												ind++;
												Output("enter the momentum value (beta): ");
												momentum = Double.parseDouble(Input(scan));
											}
										}
										optimizer = new SGDMomentum(momentum);
										break;
									case "adagrad":
										optimizer = new AdaGrad();
										break;
									case "rmsprop":
										// check if custom arguments are provided
										double decayRate = 0.9;
										if (s.length >= 8) {
											if (s[7].equals("custom")) {
												// use custom decay rate
												ind++;
												Output("enter the decay rate (rho): ");
												decayRate = Double.parseDouble(Input(scan));
											}
										}
										optimizer = new RMSProp(decayRate);
										break;
									case "adam":
										// check if custom arguments are provided
										double beta1 = 0.9;
										double beta2 = 0.999;
										if (s.length >= 8) {
											if (s[7].equals("custom")) {
												// use custom beta values
												ind++;
												Output("enter the beta1 value: ");
												beta1 = Double.parseDouble(Input(scan));
												Output("enter the beta2 value: ");
												beta2 = Double.parseDouble(Input(scan));
											}
										}
										optimizer = new Adam(beta1, beta2);
										break;
									default:
										optimizer = new SGD();
										break;
								}

								// set max batchSize to 1000
								int batchSize = mnistImages.length;
								if (s.length >= ind + 1) {
									batchSize = Integer.parseInt(s[ind]);
									ind++;
								}
								// convert labels to expected outputs
								double decay = 0;
								if (s.length >= ind + 1) {
									decay = Double.parseDouble(s[ind]);
									ind++;
								}
								double clipThreshold = 1;
								if (s.length >= ind + 1) {
									clipThreshold = Double.parseDouble(s[ind]);
									ind++;
								}
								nn.clipThreshold = clipThreshold;
								// since mnist is a classification model, display accuracy as we go
								nn.displayAccuracy = true;
								String lossFunctionStr = s[5];
								lossFunctionStr = lossFunctionStr.equals("cce") ? "categorical_crossentropy"
										: lossFunctionStr.equals("bce") ? "binary_crossentropy" : lossFunctionStr;
								LossFunction lossFunction = null;
								switch (lossFunctionStr) {
									case "mse":
										lossFunction = new MeanSquaredError();
										break;
									case "sse":
										lossFunction = new SumSquaredError();
										break;
									case "categorical_crossentropy":
										lossFunction = new CategoricalCrossentropy();
										break;
									case "mae":
										lossFunction = new MeanAbsoluteError();
										break;
									case "binary_crossentropy":
										lossFunction = new BinaryCrossentropy();
										break;
									default:
										lossFunction = new MeanSquaredError();
										break;
								}
								int epochs = Integer.parseInt(s[3]);
								ChartUpdater chartUpdater = new ChartUpdater(epochs);
								// data split into training and testing
								double splitRatio = Double.parseDouble(s[2]);
								splitRatio = Math.min(1, Math.max(1.0 / mnistImages.length, splitRatio));
								double[][] trainImages = Arrays.copyOfRange(mnistImages, 0,
										(int) (mnistImages.length * splitRatio));
								double[][] testImages = Arrays.copyOfRange(mnistImages,
										(int) (mnistImages.length * splitRatio),
										mnistImages.length);
								double[][] trainOutputs = Arrays.copyOfRange(mnistOutputs, 0,
										(int) (mnistOutputs.length * splitRatio));
								double[][] testOutputs = Arrays.copyOfRange(mnistOutputs,
										(int) (mnistOutputs.length * splitRatio),
										mnistOutputs.length);
								nn.train(trainImages, trainOutputs, testImages, testOutputs, epochs,
										Double.parseDouble(s[4]), batchSize, lossFunction, decay, optimizer,
										chartUpdater);
							}
						} else {
							// train on custom file
							File f = new File(s[1]);
							BufferedReader br = new BufferedReader(new FileReader(f));

							// get dimensions (from first line)
							StringTokenizer st = new StringTokenizer(br.readLine());
							int numCases = Integer.parseInt(st.nextToken());
							int inputSize = Integer.parseInt(st.nextToken());
							int outputSize = Integer.parseInt(st.nextToken());

							if (inputSize != nn.getTopology()[0] || outputSize != nn.getTopology()[nn.numLayers - 1]) {
								// sizes dont match
								Output("input/output sizes dont match the network");
								continue;
							}
							if (!(s[6].equals("sgd") || s[6].equals("sgdmomentum") || s[6].equals("adagrad")
									|| s[6].equals("rmsprop") || s[6].equals("adam"))) {
								// invalid loss function
								Output("invalid optimizer. choices are:\n    - sgd\n    - sgdmomentum\n    - adagrad\n    - rmsprop\n    - adam");
								continue;
							}
							int ind = 7;
							Optimizer optimizer;
							switch (s[6]) {
								case "sgd":
									optimizer = new SGD();
									break;
								case "sgdmomentum":
									// check if custom arguments are provided
									double momentum = 0.9;
									if (s.length >= 8) {
										if (s[7].equals("custom")) {
											// use custom momentum
											ind++;
											Output("enter the momentum value (beta): ");
											momentum = Double.parseDouble(Input(scan));
										}
									}
									optimizer = new SGDMomentum(momentum);
									break;
								case "adagrad":
									optimizer = new AdaGrad();
									break;
								case "rmsprop":
									// check if custom arguments are provided
									double decayRate = 0.9;
									if (s.length >= 8) {
										if (s[7].equals("custom")) {
											// use custom decay rate
											ind++;
											Output("enter the decay rate (rho): ");
											decayRate = Double.parseDouble(Input(scan));
										}
									}
									optimizer = new RMSProp(decayRate);
									break;
								case "adam":
									// check if custom arguments are provided
									double beta1 = 0.9;
									double beta2 = 0.999;
									if (s.length >= 8) {
										if (s[7].equals("custom")) {
											// use custom beta values
											ind++;
											Output("enter the beta1 value: ");
											beta1 = Double.parseDouble(Input(scan));
											Output("enter the beta2 value: ");
											beta2 = Double.parseDouble(Input(scan));
										}
									}
									optimizer = new Adam(beta1, beta2);
									break;
								default:
									optimizer = new SGD();
									break;
							}

							// set max batchSize to 1000
							int batchSize = mnistImages.length;
							if (s.length >= ind + 1) {
								batchSize = Integer.parseInt(s[ind]);
								ind++;
							}
							// convert labels to expected outputs
							double decay = 0;
							if (s.length >= ind + 1) {
								decay = Double.parseDouble(s[ind]);
								ind++;
							}
							double clipThreshold = 1;
							if (s.length >= ind + 1) {
								clipThreshold = Double.parseDouble(s[ind]);
								ind++;
							}
							nn.clipThreshold = clipThreshold;
							// parse inputs and outputs
							double[][] inputs = new double[numCases][inputSize];
							double[][] outputs = new double[numCases][outputSize];
							for (int i = 0; i < numCases; i++) {
								st = new StringTokenizer(br.readLine());
								// parse input
								for (int j = 0; j < inputSize; j++) {
									inputs[i][j] = Double.parseDouble(st.nextToken());
								}
								// skip equal sign
								st.nextToken();
								// parse output
								for (int j = 0; j < outputSize; j++) {
									outputs[i][j] = Double.parseDouble(st.nextToken());
								}
								progressBar(30, "Parsing training data", i + 1, numCases);
							}
							System.out.println();
							br.close();
							nn.displayAccuracy = false;
							String lossFunctionStr = s[5];
							lossFunctionStr = lossFunctionStr.equals("cce") ? "categorical_crossentropy"
									: lossFunctionStr.equals("bce") ? "binary_crossentropy" : lossFunctionStr;
							LossFunction lossFunction = null;
							switch (lossFunctionStr) {
								case "mse":
									lossFunction = new MeanSquaredError();
									break;
								case "sse":
									lossFunction = new SumSquaredError();
									break;
								case "categorical_crossentropy":
									lossFunction = new CategoricalCrossentropy();
									break;
								case "mae":
									lossFunction = new MeanAbsoluteError();
									break;
								case "binary_crossentropy":
									lossFunction = new BinaryCrossentropy();
									break;
								default:
									lossFunction = new MeanSquaredError();
									break;
							}
							double splitRatio = Double.parseDouble(s[2]);
							splitRatio = Math.min(1, Math.max(1.0 / inputs.length, splitRatio));
							double[][] trainInputs = Arrays.copyOfRange(inputs, 0,
									(int) (inputs.length * splitRatio));
							double[][] testInputs = Arrays.copyOfRange(inputs, (int) (inputs.length * splitRatio),
									inputs.length);
							double[][] trainOutputs = Arrays.copyOfRange(outputs, 0,
									(int) (outputs.length * splitRatio));
							double[][] testOutputs = Arrays.copyOfRange(outputs, (int) (outputs.length * splitRatio),
									outputs.length);
							nn.train(trainInputs, trainOutputs, testInputs, testOutputs, Integer.parseInt(s[3]),
									Double.parseDouble(s[4]), batchSize, lossFunction, decay, optimizer, null);
						}
					} catch (FileNotFoundException e) {
						Output("file not found");
						continue;
					} catch (Exception e) {
						Output("file parsing error");
						e.printStackTrace();
						continue;
					}
				} else if (s[0].equals("loss")) {
					if (s.length < 2) {
						Output("please specify a path to the test data and a loss function:\n    - mse\n    - mae\n    - sse\n    - categorical_crossentropy\n	   - binary_crossentropy");
						continue;
					}
					if (s[1].equals("mnist")) {
						if (!mnistInitialized) {
							Output("the mnist dataset has not yet been initialized. run 'mnist'");
							continue;
						}
						// find accuracy of mnist network
						int numCorrect = 0;
						int numCases = mnistImages.length;
						final double weightedAvg = 1.0 / (double) numCases;
						double avgLoss = 0;
						// Random r = new Random();
						for (int i = 0; i < numCases; i++) {
							int index = i;

							double[] output = nn.evaluate(mnistImages[index]);
							int prediction = indexOf(output, max(output));

							if (prediction == mnistLabels[index]) {
								numCorrect++;
							}
							avgLoss += nn.loss(output, mnistOutputs[index], new CategoricalCrossentropy())
									* weightedAvg;
							progressBar(30, "calculating", i + 1, numCases);
						}
						System.out.println();
						System.out
								.println("accuracy: " + 100 * ((double) numCorrect / numCases) + "%, loss: " + avgLoss);
						continue;
					}
					try {
						if (s.length < 3) {
							Output("please specify a path to the test data and a training function:\n    - mse\n    - mae\n    - sse\n    - categorical_crossentropy\n	   - binary_crossentropy");
							continue;
						}
						File f = new File(s[1]);
						BufferedReader br = new BufferedReader(new FileReader(f));

						// get dimensions (from first line)
						StringTokenizer st = new StringTokenizer(br.readLine());
						int numCases = Integer.parseInt(st.nextToken());
						int inputSize = Integer.parseInt(st.nextToken());
						int outputSize = Integer.parseInt(st.nextToken());

						if (inputSize != nn.getTopology()[0] || outputSize != nn.getTopology()[nn.numLayers - 1]) {
							// sizes dont match
							Output("input/output sizes dont match the network");
							continue;
						}

						// parse inputs and outputs
						double[][] inputs = new double[numCases][inputSize];
						double[][] outputs = new double[numCases][outputSize];
						for (int i = 0; i < numCases; i++) {
							st = new StringTokenizer(br.readLine());
							// parse input
							for (int j = 0; j < inputSize; j++) {
								inputs[i][j] = Double.parseDouble(st.nextToken());
							}
							// skip equal sign
							st.nextToken();
							// parse output
							for (int j = 0; j < outputSize; j++) {
								outputs[i][j] = Double.parseDouble(st.nextToken());
							}
							progressBar(30, "Parsing test data", i + 1, numCases);
						}
						System.out.println();
						br.close();
						double avgLoss = 0;
						final double weightedAvg = 1 / (double) numCases;
						for (int i = 0; i < numCases; i++) {
							double[] output = nn.evaluate(inputs[i]);
							String lossFunctionStr = s[2];
							lossFunctionStr = lossFunctionStr.equals("cce") ? "categorical_crossentropy"
									: lossFunctionStr.equals("bce") ? "binary_crossentropy" : lossFunctionStr;
							LossFunction lossFunction = null;
							switch (lossFunctionStr) {
								case "mse":
									lossFunction = new MeanSquaredError();
									break;
								case "sse":
									lossFunction = new SumSquaredError();
									break;
								case "categorical_crossentropy":
									lossFunction = new CategoricalCrossentropy();
									break;
								case "mae":
									lossFunction = new MeanAbsoluteError();
									break;
								case "binary_crossentropy":
									lossFunction = new BinaryCrossentropy();
									break;
								default:
									lossFunction = new MeanSquaredError();
									break;
							}
							double c = nn.loss(output, outputs[i], lossFunction);
							if (Double.isNaN(c)) {
								Output("nan error at input #" + i);
							}
							// Output("inputs: ");
							// printArr(inputs[i]);
							// Output("outputs: ");
							// printArr(output);
							// Output("expected: ");
							// printArr(outputs[i]);
							// Output("case " + i + " loss: " + c + " with loss function " + s[2]);
							avgLoss += c * weightedAvg;
						}
						System.out.println("loss: " + avgLoss);
					} catch (FileNotFoundException e) {
						Output("file not found");
						continue;
					} catch (Exception e) {
						Output("file parsing error");
						continue;
					}
				} else if (s[0].equals("mnist")) {
					// init mnist dataset
					if (s.length < 2) {
						Output("please specify the number of cases to import");
						continue;
					}
					mnistImages = null;
					mnistLabels = null;
					mnistOutputs = null;
					mnistInitialized = false;
					SIZE = 0;
					if (s.length > 2) {
						// check if 'augmented' modifier is present
						if (s[2].equals("augmented")) {
							// augment the dataset by applying affine transformations
							initMnist(Math.min(Integer.parseInt(s[1]), 60000), "data/train-images.idx3-ubyte",
									"data/train-labels.idx1-ubyte", true);
							continue;
						}
					}
					initMnist(Math.min(Integer.parseInt(s[1]), 60000), "data/train-images.idx3-ubyte",
							"data/train-labels.idx1-ubyte", false);
				} else if (s[0].equals("magnitude")) {
					double avgBias = 0, minBias = Double.MAX_VALUE, maxBias = Double.MIN_VALUE;
					double[][] biases = nn.getBiases();
					int numBiases = 0;
					int[] topology = nn.getTopology();
					for (int i = 0; i < biases.length; i++) {
						for (int j = 0; j < topology[i]; j++) {
							numBiases++;
						}
					}
					final double biasWeightedAvg = 1 / (double) numBiases;
					for (int i = 0; i < biases.length; i++) {
						for (int j = 0; j < topology[i]; j++) {
							avgBias += biases[i][j] * biasWeightedAvg;
							minBias = Math.min(minBias, biases[i][j]);
							maxBias = Math.max(maxBias, biases[i][j]);
						}
					}
					double avgWeight = 0, minWeight = Double.MAX_VALUE, maxWeight = Double.MIN_VALUE;
					double[][][] weights = nn.getWeights();
					int numWeights = 0;
					for (int i = 1; i < weights.length; i++) {
						for (int j = 0; j < topology[i]; j++) {
							for (int k = 0; k < topology[i - 1]; k++) {
								numWeights++;
							}
						}
					}
					final double weightWeightedAvg = 1 / (double) numWeights;
					for (int i = 1; i < weights.length; i++) {
						for (int j = 0; j < topology[i]; j++) {
							for (int k = 0; k < topology[i - 1]; k++) {
								avgWeight += weights[i][j][k] * weightWeightedAvg;
								minWeight = Math.min(minWeight, weights[i][j][k]);
								maxWeight = Math.max(maxWeight, weights[i][j][k]);
							}
						}
					}
					Output("min bias: " + minBias + "\nmax bias: " + maxBias + "\naverage bias: " + avgBias);
					Output("min weight: " + minWeight + "\nmax weight: " + maxWeight + "average weight: " + avgWeight);
				} else if (s[0].equals("help")) {
					if (s.length == 1) {
						Output("type help [command name] to get detailed usage info \ncommands: \n    - save\n    - load\n    - create\n    - init\n    - reset\n    - info\n    - evaluate\n    - exit\n    - modify\n    - regularization\n    - mutate\n    - train\n    - loss\n    - mnist\n    - magnitude\n    - help");
					} else {
						if (s[1].equals("save")) {
							Output("syntax: save [path] [object/parameters]\nsaves the current neural network to the specified file path as a java object or as a text file with parameters. text files are human readable and contain biases and activations for the input layer even though they are not used. the weights are saved as a 3D array with the first dimension being the layer (excluding the input layer), the second dimension being the neuron, and the third dimension being the incoming neuron from the previous layer");
						} else if (s[1].equals("load")) {
							Output("syntax: load [path] [object/parameters]\nloads a saved neural network from the specified path formatted as a java object or as a text file with parameters. text files are human readable and contain biases and activations for the input layer even though they are not used. the weights are saved as a 3D array with the first dimension being the layer (excluding the input layer), the second dimension being the neuron, and the third dimension being the incoming neuron from the previous layer");
						} else if (s[1].equals("create")) {
							Output("syntax: create\ncreates a custom neural network with specified properties");
						} else if (s[1].equals("init")) {
							Output("syntax: init [bias spread] [optional: weight initialization method, 'he' or 'xavier']\ninitializes current neural network parameters with random starting values and an optional weight initialization method. use 'he' for relu and 'xavier' for sigmoid/tanh");
						} else if (s[1].equals("reset")) {
							Output("syntax: reset\nresets current neural network to uninitialized");
						} else if (s[1].equals("info")) {
							Output("syntax: info [optional 'topology/activations/weights/biases/regularization']\nprints specific or general information about the current neural network.");
						} else if (s[1].equals("evaluate")) {
							Output("syntax: evaluate [optional 'mnist'] [optional mnist case #]\nevaluates the neural network for a specified input. If mnist is specified, then it will evaluate on the specified case #");
						} else if (s[1].equals("exit")) {
							Output("syntax: exit\nexits the program");
						} else if (s[1].equals("modify")) {
							Output("syntax: modify [weights/biases/activations]\nchanges a specified parameter of the current neural network");
						} else if (s[1].equals("regularization")) {
							Output("syntax: regularization [type/lambda] [value]\nsets regularization type or lambda (strength) of network.");
						} else if (s[1].equals("mutate")) {
							Output("syntax: mutate [mutation chance decimal] [variation]\nmutates neural network to simulate evolution. useful for genetic algorithms");
						} else if (s[1].equals("train")) {
							Output("syntax: train [data file path/'mnist'] [train/test split ratio, number between 0 and 1] [epochs] [learning rate] [loss function] [optimizer] [optional: 'custom', for custom optimizer hyperparameters] [optional: batch size, default=input size] [optional: decay rate, default=0] [optional: clip threshold, default=1]\ntrains neural network on specified training/test data or mnist dataset based on specified hyperparameters and optimizer\n\nloss function choices are\n    - mse\n    - mae\n    - sse\n    - categorical_crossentropy\n    - binary_crossentropy\n\noptimizer choices are:\n    - sgd\n    - sgdmomentum (momentum, default=0.9)\n    - adagrad\n    - rmsprop (decay rate, default=0.9)\n    - adam (beta1, default=0.9) (beta2, default=0.999)\n\ntraining data file must be formatted as:\n[number of cases] [input size] [output size]\n[case 1 inputs separated by spaces] = [case 1 outputs separated by spaces]\n[case 2 inputs separated by spaces] = [case 2 outputs separated by spaces]...");
						} else if (s[1].equals("loss")) {
							Output("syntax: loss [test data file path] [loss function] or loss mnist\nreturns the average cross entropy loss of the neural network for the specified dataset or the accuracy percentage for the mnist dataset. loss function choices are\n    - mse\n    - mae\n    - sse\n    - categorical_crossentropy\n    - binary_crossentropy\ntest data file must be formatted as:\n[number of cases] [input size] [output size]\n[case 1 inputs separated by spaces] = [case 1 outputs separated by spaces]\n[case 2 inputs separated by spaces] = [case 2 outputs separated by spaces]...");
						} else if (s[1].equals("help")) {
							Output("syntax: help [optional: command name]\nhelp command");
						} else if (s[1].equals("mnist")) {
							Output("syntax: mnist [# of cases] [optional 'augmented'] OR syntax: mnist test\ninitializes the mnist dataset with the specified # of cases. up to 60,000. An optional 'augmented' modifier can be specified that applies random affine transformations to the data to make training more robust. OR, allows users to test the network on custom user-drawn digits");
						} else if (s[1].equals("magnitude")) {
							Output("syntax: magnitude\ndisplays the magnitude of the network's parameters. Shows min/max/average weights and biases");
						} else {
							// unknown command
							Output(s[1] + ": command not found");
						}
					}
				} else {
					// invalid command
					Output(s[0] + ": command not found");
				}
			} catch (NullPointerException e) {
				Output("ERROR: neural network has not been initialized");
				continue;
			} catch (IndexOutOfBoundsException e) {
				Output("ERROR: input is out of allowed range");
			} catch (Exception e) {
				Output("ERROR: invalid input");
				e.printStackTrace();
				continue;
			}
		}
		scan.close();
	}

	static double max(double[] arr) {
		double m = -1;
		for (double i : arr) {
			if (i > m) {
				m = i;
			}
		}
		return m;
	}

	static int indexOf(double[] arr, double v) {
		int index = -1;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}

	static void testMnist(NeuralNetwork nn) {
		JFrame frame = new JFrame("Draw Image");
		JLabel[] labels = new JLabel[10];
		final RealTimeSoftDrawCanvas drawCanvas = new RealTimeSoftDrawCanvas();

		frame = new JFrame("Draw Image");
		drawCanvas.clearCanvas();
		JButton clearButton = new JButton("Clear");

		clearButton.addActionListener(e -> drawCanvas.clearCanvas());

		JPanel panel = new JPanel();
		panel.setLayout(new BorderLayout());

		JPanel content = new JPanel();
		GridLayout contentLayout = new GridLayout(1, 2);
		contentLayout.setHgap(20);
		content.setLayout(contentLayout);
		content.add(drawCanvas);

		JPanel labelsPanel = new JPanel();
		// list of labels from 0-9 with listed probabilities
		labelsPanel.setLayout(new GridLayout(10, 1));
		Font font = new Font("Arial", Font.PLAIN, 20);
		for (int i = 0; i < 10; i++) {
			JLabel l = new JLabel(i + ": 0.00%");
			l.setFont(font);
			labelsPanel.add(l);
			labels[i] = l;
		}
		// add padding on right
		content.add(labelsPanel);

		JPanel controls = new JPanel();
		controls.add(clearButton);
		panel.add(content, BorderLayout.CENTER);
		panel.add(controls, BorderLayout.SOUTH);
		frame.setContentPane(panel);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setVisible(true);

		while (frame.isVisible()) {
			// evaluate the image and display the output
			double[][] inputRaw = drawCanvas.getPixelArray();
			// flatten the array
			double[] input = new double[inputRaw.length * inputRaw[0].length];
			for (int i = 0; i < inputRaw.length; i++) {
				for (int j = 0; j < inputRaw[0].length; j++) {
					input[i * inputRaw[0].length + j] = inputRaw[i][j];
				}
			}
			double[] output = nn.evaluate(input);
			// find the highest probability
			int maxIndex = 0;
			for (int i = 0; i < 10; i++) {
				if (output[i] > output[maxIndex]) {
					maxIndex = i;
				}
			}
			for (int i = 0; i < 10; i++) {
				// highlight the highest probability
				if (i == maxIndex) {
					labels[i].setForeground(Color.RED);
				} else {
					labels[i].setForeground(Color.BLACK);
				}
				labels[i].setText(i + ": " + String.format("%.2f", output[i] * 100) + "%");
			}
		}
		// the realtimesoftdrawcanvas object has a timer that needs to be stopped
		drawCanvas.stopTimer();
	}

	static void initMnist(int numCases, String dataFilePath, String labelFilePath, boolean doAugment) {
		mnistInitialized = true;

		try {
			SIZE = numCases;

			DataInputStream dataInputStream = new DataInputStream(
					new BufferedInputStream(new FileInputStream(dataFilePath)));
			int magicNumber = dataInputStream.readInt();
			int numberOfItems = dataInputStream.readInt();
			int nRows = dataInputStream.readInt();
			int nCols = dataInputStream.readInt();

			mnistLabels = new int[SIZE];
			mnistOutputs = new double[SIZE][10];
			mnistImages = new double[SIZE][784];

			DataInputStream labelInputStream = new DataInputStream(
					new BufferedInputStream(new FileInputStream(labelFilePath)));
			int labelMagicNumber = labelInputStream.readInt();
			int numberOfLabels = labelInputStream.readInt();

			for (int i = 0; i < SIZE; i++) {
				mnistLabels[i] = (labelInputStream.readUnsignedByte());
				mnistOutputs[i][mnistLabels[i]] = 1;
				if (doAugment) {
					double[] imgFlat = new double[nRows * nCols];
					for (int r = 0; r < nRows * nCols; r++) {
						imgFlat[r] = dataInputStream.readUnsignedByte();
					}
					double[][] img = new double[nRows][nCols];
					for (int r = 0; r < nRows; r++) {
						for (int c = 0; c < nCols; c++) {
							img[r][c] = imgFlat[r * nCols + c] / 255.0;
						}
					}
					// augment image
					img = scale(img, (Math.random() * 0.4 + 0.65));
					img = shift(img, (int) (Math.random() * 10 - 5), (int) (Math.random() * 10 - 5));
					img = rotate(img, (int) (Math.random() * 50 - 25));
					// flatten image
					for (int r = 0; r < nRows; r++) {
						for (int c = 0; c < nCols; c++) {
							mnistImages[i][r * nCols + c] = img[r][c];
						}
					}
				} else {
					for (int r = 0; r < nRows * nCols; r++) {
						mnistImages[i][r] = dataInputStream.readUnsignedByte() / 255.0;
					}
				}
				progressBar(30, "parsing MNIST", i + 1, SIZE);
			}
			System.out.println();
			dataInputStream.close();
			labelInputStream.close();
			// showImage(mnistImages[0], 28, 28);
			// printArr(mnistOutputs[0]);
			// System.out.println(mnistLabels[0]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void showImage(double[] image, int width, int height) {
		String filled = "██";
		String unfilled = "░░";
		for (int i = 0; i < height; i++) {
			String line = "";
			for (int j = 0; j < width; j++) {
				line += image[width * i + j] >= 0.5 ? filled : unfilled;
			}
			Output(line);
		}
	}

	public static void showImage(double[][] image) {
		String filled = "██";
		String unfilled = "░░";
		for (int i = 0; i < image.length; i++) {
			String line = "";
			for (int j = 0; j < image[0].length; j++) {
				line += image[i][j] >= 0.5 ? filled : unfilled;
			}
			Output(line);
		}
	}

	public static double[][] shift(double[][] image, int shiftX, int shiftY) {
		int width = image.length;
		int height = image[0].length;
		double[][] shiftedImage = new double[width][height];

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int newX = x + shiftX;
				int newY = y + shiftY;
				if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
					shiftedImage[newX][newY] = image[x][y];
				}
			}
		}
		return shiftedImage;
	}

	public static double[][] rotate(double[][] image, double angle) {
		int width = image.length;
		int height = image[0].length;
		double[][] rotatedImage = new double[width][height];
		double radians = Math.toRadians(angle);
		int centerX = width / 2;
		int centerY = height / 2;

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int newX = (int) ((x - centerX) * Math.cos(radians) - (y - centerY) * Math.sin(radians) + centerX);
				int newY = (int) ((x - centerX) * Math.sin(radians) + (y - centerY) * Math.cos(radians) + centerY);
				if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
					rotatedImage[newX][newY] = image[x][y];
				}
			}
		}
		return rotatedImage;
	}

	public static double[][] scale(double[][] image, double scaleFactor) {
		int width = image.length;
		int height = image[0].length;
		double[][] scaledImage = new double[width][height];

		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				// Calculate the original coordinates
				double origX = (x - width / 2.0) / scaleFactor + width / 2.0;
				double origY = (y - height / 2.0) / scaleFactor + height / 2.0;

				// Interpolate pixel value
				scaledImage[x][y] = interpolatePixel(image, origX, origY);
			}
		}
		return scaledImage;
	}

	private static double interpolatePixel(double[][] image, double x, double y) {
		int x1 = (int) Math.floor(x);
		int y1 = (int) Math.floor(y);
		int x2 = x1 + 1;
		int y2 = y1 + 1;

		if (x1 >= 0 && x2 < image.length && y1 >= 0 && y2 < image[0].length) {
			double Q11 = image[x1][y1];
			double Q12 = image[x1][y2];
			double Q21 = image[x2][y1];
			double Q22 = image[x2][y2];

			double R1 = ((x2 - x) / (x2 - x1)) * Q11 + ((x - x1) / (x2 - x1)) * Q21;
			double R2 = ((x2 - x) / (x2 - x1)) * Q12 + ((x - x1) / (x2 - x1)) * Q22;

			return ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2;
		} else {
			return 0;
		}
	}

	public static void sleep(int milliseconds) {
		try {
			Thread.sleep(milliseconds);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	static void progressBar(int width, String title, int current, int total) {
		String filled = "█";
		String unfilled = "░";
		double fill = (double) current / total;
		if (fill >= 0 && fill <= 1) {
			// set progress bar
			int fillAmount = (int) Math.ceil(fill * width);
			StringBuilder bar = new StringBuilder();
			bar.append(title).append(": ").append(filled.repeat(fillAmount)).append(unfilled.repeat(width - fillAmount))
					.append(" ").append(current).append("/").append(total).append("\r");
			System.out.print(bar.toString());
		}
	}

	static boolean contains(String[] s, String value) {
		for (int i = 0; i < s.length; i++) {
			if (s[i].equals(value)) {
				return true;
			}
		}
		return false;
	}

	static void Clear() {
		Output("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		Output("Neural Network Console (type 'help' or 'exit' if stuck)");
		Output("=-=-=-=-=-=-=-=-=-=-=-=-=-=");
	}

	static void Output(String msg) {
		System.out.println("" + msg);
	}

	static String Input(Scanner scan) {
		System.out.print(">> ");
		return scan.nextLine();
	}

	static void printArr(int[] arr) {
		String print = "[";
		for (int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static void printArr(double[] arr) {
		String print = "[";
		for (int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static void printArr(String[] arr) {
		String print = "[";
		for (int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static String returnArr(double[] arr) {
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
}