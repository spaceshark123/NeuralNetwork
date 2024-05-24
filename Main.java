import java.util.*;
import java.lang.*;
import java.io.*;

class Main {
	public static int SIZE = 1000;
	public static double[][] mnistOutputs;
	public static int[] mnistLabels;
	public static double[][] mnistImages;
	public static boolean mnistInitialized = false;
	
  	public static void main(String[] args) {
		//load?
		Scanner scan = new Scanner(System.in);
		String path = "SavedNetwork";
		NeuralNetwork nn = new NeuralNetwork();
		Clear();

		while(true) {
			String[] s = Input(scan).split(" ");
			try {
				if(s[0].equals("load")) {
					if(s.length > 1) {
						//path argument is included
						Output("loading...");
						nn = NeuralNetwork.Load(s[1]);
					} else {
						//use default path
						//nn = NeuralNetwork.Load(path);
						Output("please specify a path.");
					} 
				} else if(s[0].equals("save")) {
					if(s.length > 1) {
						Output("saving...");
						//path argument is included
						NeuralNetwork.Save(nn, s[1]);
					} else {
						//use default path
						//NeuralNetwork.Save(nn, path);
						Output("please specify a path.");
					}
				} else if(s[0].equals("exit")) {
					Output("exiting...");
					break;
				} else if(s[0].equals("info")) {
					//print info about neural network
					if(s.length == 1) {
						System.out.println(nn);
					} else {
						if(s[1].equals("activations")) {
							System.out.print("activations: ");
							printArr(nn.GetActivations());
						} else if(s[1].equals("topology")) {
							System.out.print("topology: ");
							printArr(nn.GetTopology());
						} else if(s[1].equals("regularization")) {
							System.out.println("regularization: " + nn.GetRegularizationType().toString() + " lambda: " + nn.GetRegularizationLambda());
						} else if(s[1].equals("biases")) {
							String print = "";
							print += "\nBiases:\n";
							for(int i = 0; i < nn.numLayers; i++) {
								print += "Layer " + (i+1) + ": " + returnArr(Arrays.copyOfRange(nn.GetBiases()[i],0,nn.GetTopology()[i])) + "\n";
							}
							Output(print);
						} else if(s[1].equals("weights")) {
							String print = "";
							print += "\nWeights:\n";
							for(int i = 1; i < nn.numLayers; i++) {
								for(int j = 0; j < nn.GetTopology()[i]; j++) {
									//each neuron
									print += "    Neuron " + (j+1) + " of Layer " + (i+1) + " Weights: \n" + returnArr(Arrays.copyOfRange(nn.GetWeights()[i][j],0,nn.GetTopology()[i-1])) + "\n"; 
								}
							}
							Output(print);
						} else {
							Output("not a valid property. choices are:\n    - activations\n    - topology\n    - biases\n    - weights");
						}
					}
				} else if(s[0].equals("clear")) {
					//clear console
					Clear();
					Output("cleared console");
				} else if(s[0].equals("create")) {
					//create a new neural network
					Output("Number of layers (at least 2): ");
					int[] topology = new int[Integer.parseInt(Input(scan))];
					Output("Size of each layer (separated by spaces): ");
					String[] ls = Input(scan).split(" ");
					if(ls.length != topology.length) {
						Output("mismatch in number of layers");
						continue;
					}
					for(int i = 0; i < ls.length; i++) {
						topology[i] = Integer.parseInt(ls[i]);
					}
					Output("Activations for each layer (separated by spaces): ");
					Output("Choices: \n    - linear\n    - sigmoid\n    - tanh\n    - relu\n    - binary\n    - softmax");
					ls = Input(scan).split(" ");
					if(ls.length != topology.length) {
						Output("mismatch in number of layers");
						continue;
					}
					nn = new NeuralNetwork(topology, ls);
					Output("Created neural network");
				} else if(s[0].equals("init")) {
					//initialize neural network
					if(s.length >= 2) {
						Output("initializing weights and biases...");
						nn.Init(Double.parseDouble(s[1]));
						if(s.length >= 3) {
							nn.Init(s[2], Double.parseDouble(s[1]));
						}
					} else {
						Output("please specify a bias spread and optionally a weight initialization method ('he' or 'xavier')");
						continue;
					}
				} else if(s[0].equals("regularization")) {
					//modify regularization
					if(s.length < 2) {
						//no argument is provided
						Output("please specify a valid property to modify: \n    - type\n    - lambda");
						continue;
					}
					if(s[1].equals("type")) {
						if(s.length < 3) {
							//no type is provided
							Output("please specify a regularization to apply (not case sensitive): \n    - none\n    - L1\n    - L2");
							continue;
						}
						if(s[2].equalsIgnoreCase("none")) {
							nn.SetRegularizationType(NeuralNetwork.RegularizationType.NONE);
							Output("applying regularization...");
						} else if(s[2].equalsIgnoreCase("L1")) {
							nn.SetRegularizationType(NeuralNetwork.RegularizationType.L1);
							Output("applying regularization...");
						} else if(s[2].equalsIgnoreCase("L2")) {
							nn.SetRegularizationType(NeuralNetwork.RegularizationType.L2);
							Output("applying regularization...");
						} else {
							//invalid type is provided
							Output("please specify a valid regularization to apply (not case sensitive): \n    - none\n    - L1\n    - L2");
							continue;
						}
					} else if(s[1].equals("lambda")) {
						if(s.length < 3) {
							//no type is provided
							Output("please specify a regularization lambda (strength) to set.");
							continue;
						}
						nn.SetRegularizationLambda(Double.parseDouble(s[2]));
						Output("setting lambda...");
					} else {
						//invalid argument is provided
						Output("please specify a valid property to modify: \n    - type\n    - lambda");
						continue;
					}
				} else if(s[0].equals("evaluate")) {
					if(s.length > 2) {
						if(s[1].equals("mnist")) {
							if(!mnistInitialized) {
								Output("the mnist dataset has not yet been initialized. run 'mnist'");
								continue;
							}
							int index = Integer.parseInt(s[2]);
							showImage(mnistImages[index], 28, 28);
							double[] output = nn.Evaluate(mnistImages[index]);
							Output("predicted: " + indexOf(output, max(output)));
							Output("actual: " + mnistLabels[index]);
							System.out.print("output: ");
							printArr(output);
							continue;
						}
					}
					Output(nn.GetTopology()[0] + " input(s) (separated by spaces): ");
					String[] ls = Input(scan).split(" ");
					double[] input = new double[nn.GetTopology()[0]];
					if(ls.length != nn.GetTopology()[0]) {
						Output("mismatch in number of inputs");
						continue;
					}
					for(int i = 0; i < ls.length; i++) {
						input[i] = Double.parseDouble(ls[i]);
					}
					printArr(nn.Evaluate(input));
				} else if(s[0].equals("reset")) {
					Output("resetting network...");
					nn = new NeuralNetwork();
				} else if(s[0].equals("modify")) {
					if(s.length > 1) {
						if(s[1].equals("activations")) {
							Output("enter the layer to modify (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if(layer >= nn.numLayers || layer < 0) {
								//not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the new activation: ");
							Output("Choices: \n    - linear\n    - sigmoid\n    - tanh\n    - relu\n    - binary\n    - softmax");
							nn.SetActivation(layer, Input(scan));
							Output("making modification...");
						} else if(s[1].equals("weights")) {
							Output("enter the layer of the end neuron (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if(layer >= nn.numLayers || layer < 0) {
								//not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the neuron # of the end neuron (1 is first neuron): ");
							int end = Integer.parseInt(Input(scan)) - 1;
							if(end >= nn.GetTopology()[layer] || end < 0) {
								//not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the neuron # of the start neuron from the previous layer (1 is first neuron): ");
							int start = Integer.parseInt(Input(scan)) - 1;
							if(start >= nn.GetTopology()[layer-1] || start < 0) {
								//not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the new weight: ");
							nn.SetWeight(layer, end, start, Double.parseDouble(Input(scan)));
							Output("setting weight...");
						} else if(s[1].equals("biases")) {
							Output("enter the layer of the neuron (1 is first layer): ");
							int layer = Integer.parseInt(Input(scan)) - 1;
							if(layer >= nn.numLayers || layer < 0) {
								//not a valid layer #
								Output("not a valid layer");
								continue;
							}
							Output("enter the neuron # (1 is first neuron): ");
							int end = Integer.parseInt(Input(scan)) - 1;
							if(end >= nn.GetTopology()[layer] || end < 0) {
								//not a valid layer #
								Output("not a valid neuron #");
								continue;
							}
							Output("enter the new bias: ");
							nn.SetBias(layer, end, Double.parseDouble(Input(scan)));
							Output("setting bias...");
						} else {
							//modify argument isnt valid
							Output("please specify a valid property to modify: \n    - activations\n    - weights\n    - biases");
							continue;
						}
					} else {
						//no modify argument is provided
						Output("please specify a valid property to modify: \n    - activations\n    - weights\n    - biases");
						continue;
					}
				} else if(s[0].equals("mutate")) {
					if(s.length >= 3) {
						Output("mutating...");
						nn.Mutate(Double.parseDouble(s[1]), Double.parseDouble(s[2]));
					} else {
						Output("please specify the mutation chance (decimal) and variation");
						continue;
					}
				} else if(s[0].equals("train")) {
					if(s.length < 5) {
						Output("please specify the following:\n    - path to the training set\n    - number of epochs\n    - learning rate\n    - loss function\n    - batch size (optional)\n    - decay rate (optional)\n    - clip threshold (optional)\n    - momentum (optional)");
						continue;
					}
					//trainset file must be formatted:
					//first line has 3 numbers specifying # cases, input and output size
					//every line is a separate training case
					//on each line, input is separated by spaces, then equal sign, then output separated by spaces
					try {
						if(s[1].equals("mnist")) {
							//train on mnist
							if(!mnistInitialized) {
								Output("the mnist dataset has not yet been initialized. run 'mnist'");
								continue;
							} else {
								if(!(s[4].equals("mse") || s[4].equals("categorical_crossentropy") || s[4].equals("cce") || s[4].equals("sse"))) {
									//invalid loss function
									Output("invalid loss function. choices are:\n    - mse\n    - sse\n    - categorical_crossentropy");
									continue;
								}

								//set max batchSize to 1000
								int batchSize = mnistImages.length;
								if(s.length >= 6) {
									batchSize = Integer.parseInt(s[5]);
								}
								//convert labels to expected outputs
								double decay = 0;
								if(s.length >= 7) {
									decay = Double.parseDouble(s[6]);
								}
								double clipThreshold = 1;
								if(s.length >= 8) {
									clipThreshold = Double.parseDouble(s[7]);
								}
								nn.clipThreshold = clipThreshold;
								double momentum = 0.1;
								if(s.length >= 9) {
									momentum = Double.parseDouble(s[8]);
								}
								//since mnist is a classification model, display accuracy as we go
								nn.displayAccuracy = true;
								String lossFunction = s[4];
								lossFunction = lossFunction.equals("cce") ? "categorical_crossentropy" : lossFunction;
								int epochs = Integer.parseInt(s[2]);
								ChartUpdater chartUpdater = new ChartUpdater(epochs);
								nn.Train(mnistImages, mnistOutputs, epochs, Double.parseDouble(s[3]), batchSize, lossFunction, decay, momentum, chartUpdater);
							}
						} else {
							//train on custom file
							File f = new File(s[1]);
							BufferedReader br = new BufferedReader(new FileReader(f));
		
							//get dimensions (from first line)
							StringTokenizer st = new StringTokenizer(br.readLine());
							int numCases = Integer.parseInt(st.nextToken());
							int inputSize = Integer.parseInt(st.nextToken());
							int outputSize = Integer.parseInt(st.nextToken());
		
							if(inputSize != nn.GetTopology()[0] || outputSize != nn.GetTopology()[nn.numLayers-1]) {
								//sizes dont match
								Output("input/output sizes dont match the network");
								continue;
							}
							if(!(s[4].equals("mse") || s[4].equals("categorical_crossentropy") || s[4].equals("cce") || s[4].equals("sse"))) {
								//invalid loss function
								Output("invalid loss function. choices are:\n    - mse\n    - sse\n    - categorical_crossentropy");
								continue;
							}
							//parse inputs and outputs
							double[][] inputs = new double[numCases][inputSize];
							double[][] outputs = new double[numCases][outputSize];
							for(int i = 0; i < numCases; i++) {
								st = new StringTokenizer(br.readLine());
								//parse input
								for(int j = 0; j < inputSize; j++) {
									inputs[i][j] = Double.parseDouble(st.nextToken());
								}
								//skip equal sign
								st.nextToken();
								//parse output
								for(int j = 0; j < outputSize; j++) {
									outputs[i][j] = Double.parseDouble(st.nextToken());
								}
								progressBar(30, "Parsing training data", i+1, numCases);
							}
							System.out.println();
		
							int batchSize = inputs.length;
							if(s.length >= 6) {
								batchSize = Integer.parseInt(s[5]);
							}
							double decay = 0;
							if(s.length >= 7) {
								decay = Double.parseDouble(s[6]);
							}
						    double clipThreshold = 1;
							if(s.length >= 8) {
								clipThreshold = Double.parseDouble(s[7]);
							}
							nn.clipThreshold = clipThreshold;
							double momentum = 0.1;
							if(s.length >= 9) {
								momentum = Double.parseDouble(s[8]);
							}
							nn.displayAccuracy = false;
							String lossFunction = s[4];
							lossFunction = lossFunction.equals("cce") ? "categorical_crossentropy" : lossFunction;
							nn.Train(inputs, outputs, Integer.parseInt(s[2]), Double.parseDouble(s[3]), batchSize, lossFunction, decay, momentum, null);
						}
					} catch(FileNotFoundException e) {
						Output("file not found");
						continue;
					} catch(Exception e) {
						Output("file parsing error");
						e.printStackTrace(); 
						continue;
					}
				} else if(s[0].equals("cost")) {
					if(s.length < 2) {
						Output("please specify a path to the test data and a training function:\n    - mse\n    - sse\n    - categorical_crossentropy");
						continue;
					}
					if(s[1].equals("mnist")) {
						if(!mnistInitialized) {
							Output("the mnist dataset has not yet been initialized. run 'mnist'");
							continue;
						}
						//find accuracy of mnist network
						int numCorrect = 0;
						int numCases = mnistImages.length;
						final double weightedAvg = 1.0 / (double)numCases;
						double avgCost = 0;
						//Random r = new Random();
						for(int i = 0; i < numCases; i++) {
							int index = i;

							double[] output = nn.Evaluate(mnistImages[index]);
							int prediction = indexOf(output, max(output));

							if(prediction == mnistLabels[index]) {
								numCorrect++;
							}
							avgCost += nn.Cost(output, mnistOutputs[index], "categorical_crossentropy") * weightedAvg;
							progressBar(30, "calculating", i+1, numCases);
						}
						System.out.println();
						System.out.println("accuracy: " + 100*((double) numCorrect / numCases) + "%, cost: " + avgCost);
						continue;
					}
					try {
						if(s.length < 3) {
							Output("please specify a path to the test data and a training function:\n    - mse\n    - sse\n    - categorical_crossentropy");
							continue;
						}
						File f = new File(s[1]);
						BufferedReader br = new BufferedReader(new FileReader(f));
		
						//get dimensions (from first line)
						StringTokenizer st = new StringTokenizer(br.readLine());
						int numCases = Integer.parseInt(st.nextToken());
						int inputSize = Integer.parseInt(st.nextToken());
						int outputSize = Integer.parseInt(st.nextToken());
		
						if(inputSize != nn.GetTopology()[0] || outputSize != nn.GetTopology()[nn.numLayers-1]) {
							//sizes dont match
							Output("input/output sizes dont match the network");
							continue;
						}
		
						//parse inputs and outputs
						double[][] inputs = new double[numCases][inputSize];
						double[][] outputs = new double[numCases][outputSize];
						for(int i = 0; i < numCases; i++) {
							st = new StringTokenizer(br.readLine());
							//parse input
							for(int j = 0; j < inputSize; j++) {
								inputs[i][j] = Double.parseDouble(st.nextToken());
							}
							//skip equal sign
							st.nextToken();
							//parse output
							for(int j = 0; j < outputSize; j++) {
								outputs[i][j] = Double.parseDouble(st.nextToken());
							}
							progressBar(30, "Parsing test data", i+1, numCases);
						}
						System.out.println();
						double avgCost = 0;
						final double weightedAvg = 1/(double)numCases;
						for(int i = 0; i < numCases; i++) {
							double[] output = nn.Evaluate(inputs[i]);
							double c = nn.Cost(output, outputs[i], s[2]);
							if(Double.isNaN(c)) {
								Output("nan error at input #" + i);
							}
							// Output("inputs: ");
							// printArr(inputs[i]);
							// Output("outputs: ");
							// printArr(output);
							// Output("expected: ");
							// printArr(outputs[i]);
							// Output("case " + i + " cost: " + c + " with loss function " + s[2]);
							avgCost += c * weightedAvg;
						}
						System.out.println("cost: " + avgCost);
					} catch(FileNotFoundException e) {
						Output("file not found");
						continue;
					} catch(Exception e) {
						Output("file parsing error");
						continue;
					}
				} else if(s[0].equals("mnist")) {
					//init mnist dataset
					if(s.length < 2) {
						Output("please specify the number of cases to import");
						continue;
					}
					mnistImages = null;
					mnistLabels = null;
					mnistOutputs = null;
					mnistInitialized = false;
					SIZE = 0;
					initMnist(Math.min(Integer.parseInt(s[1]), 60000), "data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
				} else if(s[0].equals("magnitude")) {
					double avgBias = 0, minBias = Double.MAX_VALUE, maxBias = Double.MIN_VALUE;	
					double[][] biases = nn.GetBiases();
					int numBiases = 0;
					int[] topology = nn.GetTopology();
					for(int i = 0; i < biases.length; i++) {
						for(int j = 0; j < topology[i]; j++) {
							numBiases++;
						}
					}
					final double biasWeightedAvg = 1/(double)numBiases;
					for(int i = 0; i < biases.length; i++) {
						for(int j = 0; j < topology[i]; j++) {
							avgBias += biases[i][j] * biasWeightedAvg;
							minBias = Math.min(minBias, biases[i][j]);
							maxBias = Math.max(maxBias, biases[i][j]);
						}
					}
					double avgWeight = 0, minWeight = Double.MAX_VALUE, maxWeight = Double.MIN_VALUE;	
					double[][][] weights = nn.GetWeights();
					int numWeights = 0;
					for(int i = 1; i < weights.length; i++) {
						for(int j = 0; j < topology[i]; j++) {
							for(int k = 0; k < topology[i-1]; k++) {
								numWeights++;
							}
						}
					}
					final double weightWeightedAvg = 1/(double)numWeights;
					for(int i = 1; i < weights.length; i++) {
						for(int j = 0; j < topology[i]; j++) {
							for(int k = 0; k < topology[i-1]; k++) {
								avgWeight += weights[i][j][k] * weightWeightedAvg;
								minWeight = Math.min(minWeight, weights[i][j][k]);
								maxWeight = Math.max(maxWeight, weights[i][j][k]);
							}
						}
					}
					Output("min bias: " + minBias + "\nmax bias: " + maxBias + "\naverage bias: " + avgBias);
					Output("min weight: " + minWeight + "\nmax weight: " + maxWeight + "average weight: " + avgWeight);
				} else if(s[0].equals("help")) {
					if(s.length == 1) {
						Output("type help [command name] to get detailed usage info \ncommands: \n    - save\n    - load\n    - create\n    - init\n    - reset\n    - info\n    - evaluate\n    - exit\n    - modify\n    - regularization\n    - mutate\n    - train\n    - cost\n    - mnist\n    - magnitude\n    - help");
					} else {
						if(s[1].equals("save")) {
							Output("syntax: save [path]\nsaves the current neural network to the specified file path");
						} else if(s[1].equals("load")) {
							Output("syntax: load [path]\nloads a saved neural network from the specified path");
						} else if(s[1].equals("create")) {
							Output("syntax: create\ncreates a custom neural network with specified properties");
						} else if(s[1].equals("init")) {
							Output("syntax: init [bias spread] [optional: weight initialization method, 'he' or 'xavier']\ninitializes current neural network parameters with random starting values and an optional weight initialization method. use 'he' for relu and 'xavier' for sigmoid/tanh");
						} else if(s[1].equals("reset")) {
							Output("syntax: reset\nresets current neural network to uninitialized");
						} else if(s[1].equals("info")) {
							Output("syntax: info [optional 'topology/activations/weights/biases/regularization']\nprints specific or general information about the current neural network.");
						} else if(s[1].equals("evaluate")) {
							Output("syntax: evaluate [optional 'mnist'] [optional mnist case #]\nevaluates the neural network for a specified input. If mnist is specified, then it will evaluate on the specified case #");
						} else if(s[1].equals("exit")) {
							Output("syntax: exit\nexits the program");
						} else if(s[1].equals("modify")) {
							Output("syntax: modify [weights/biases/activations]\nchanges a specified parameter of the current neural network");
						} else if(s[1].equals("regularization")) {
							Output("syntax: regularization [type/lambda] [value]\nsets regularization type or lambda (strength) of network.");
						} else if(s[1].equals("mutate")) {
							Output("syntax: mutate [mutation chance decimal] [variation]\nmutates neural network to simulate evolution. useful for genetic algorithms");
						} else if(s[1].equals("train")) {
							Output("syntax: train [training data file path/'mnist'] [epochs] [learning rate] [loss function] [optional: batch size, default=input size] [optional: decay rate, default=0] [optional: clip threshold, default=1] [optional: momentum, default=0.1]\ntrains neural network on specified training data or mnist dataset based on specified hyperparameters. loss function choices are\n    - mse\n    - sse\n    - categorical_crossentropy\ntraining data file must be formatted as:\n[number of cases] [input size] [output size]\n[case 1 inputs separated by spaces] = [case 1 outputs separated by spaces]\n[case 2 inputs separated by spaces] = [case 2 outputs separated by spaces]...");
						} else if(s[1].equals("cost")) {
							Output("syntax: cost [test data file path] [loss function] or cost mnist\nreturns the average cost of the neural network for the specified dataset or the accuracy percentage for the mnist dataset. loss function choices are\n    - mse\n    - sse\n    - categorical_crossentropy\ntest data file must be formatted as:\n[number of cases] [input size] [output size]\n[case 1 inputs separated by spaces] = [case 1 outputs separated by spaces]\n[case 2 inputs separated by spaces] = [case 2 outputs separated by spaces]...");
						} else if(s[1].equals("help")) {
							Output("syntax: help [optional: command name]\nhelp command");
						} else if(s[1].equals("mnist")) {
							Output("syntax: mnist [# of cases]\ninitializes the mnist dataset with the specified # of cases. up to 60,000");
						} else if(s[1].equals("magnitude")) {
							Output("syntax: magnitude\ndisplays the magnitude of the network's parameters. Shows min/max/average weights and biases");
						} else {
							//unknown command
							Output(s[1] + ": command not found");
						}
					}
				} else {
					//invalid command
					Output(s[0] + ": command not found");
				}
			} catch(NullPointerException e) {
				Output("ERROR: neural network has not been initialized");
				continue;
			} catch(IndexOutOfBoundsException e) {
				Output("ERROR: input is out of allowed range");
			} catch(Exception e) {
				Output("ERROR: invalid input");
				e.printStackTrace();
				continue;
			}
		}
		scan.close();
	}

	static double max(double[] arr) {
		double m = -1;
		for(double i : arr) {
			if(i > m) {
				m = i;
			}
		}
		return m;
	}

	static int indexOf(double[] arr, double v) {
		int index = -1;
		for(int i = 0; i < arr.length; i++) {
			if(arr[i] == v) {
				index = i;
				return index;
			}
		}
		return index;
	}

	static void initMnist(int numCases, String dataFilePath, String labelFilePath) {
		mnistInitialized = true;
		
		try {
			SIZE = numCases;
			
			DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
	        int magicNumber = dataInputStream.readInt();
	        int numberOfItems = dataInputStream.readInt();
	        int nRows = dataInputStream.readInt();
	        int nCols = dataInputStream.readInt();
	
			mnistLabels = new int[SIZE];
			mnistOutputs = new double[SIZE][10];
			mnistImages = new double[SIZE][784];
	
	        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
	        int labelMagicNumber = labelInputStream.readInt();
	        int numberOfLabels = labelInputStream.readInt();
	
	        for(int i = 0; i < SIZE; i++) {
	            mnistLabels[i] = (labelInputStream.readUnsignedByte());
				mnistOutputs[i][mnistLabels[i]] = 1;
	            for (int r = 0; r < nRows*nCols; r++) {
	                mnistImages[i][r] = dataInputStream.readUnsignedByte();
	            }
				progressBar(30, "parsing MNIST", i+1, SIZE);
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

	static void showImage(double[] image, int width, int height) {
		String filled = "██";
		String unfilled = "░░";
		for(int i = 0; i < height; i++) {
			String line = "";
			for(int j = 0; j < width; j++) {
				line += image[width*i+j] >= 0.5 ? filled : unfilled;
			}
			Output(line);
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
			//set progress bar
			int fillAmount = (int) Math.ceil(fill * width);
			StringBuilder bar = new StringBuilder();
			bar.append(title).append(": ").append(filled.repeat(fillAmount)).append(unfilled.repeat(width - fillAmount)).append(" ").append(current).append("/").append(total).append("\r");
			System.out.print(bar.toString());
		}
	}

	static boolean contains(String[] s, String value) {
		for(int i = 0; i < s.length; i++) {
			if(s[i].equals(value)) {
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
		for(int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static void printArr(double[] arr) {
		String print = "[";
		for(int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static void printArr(String[] arr) {
		String print = "[";
		for(int i = 0; i < arr.length - 1; i++) {
			print += arr[i] + ", ";
		}
		print += arr[arr.length - 1] + "]";
		System.out.println(print);
	}

	static String returnArr(double[] arr) {
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
}