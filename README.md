

# Neural Network Implementation in Java

This project provides a flexible and extensible implementation of a stochastic gradient descent feedforward neural network in Java, wrapped up in a console user interface with a realtime accuracy graph visualizer while training. The neural network is designed to be easy to use and customize, making it a valuable tool for various machine learning and deep learning tasks.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Internal Usage](#internal-usage)
- [Program Usage](#program-usage)
- [Examples](#examples)

## Overview

Neural networks are a fundamental building block of modern machine learning and artificial intelligence. This Java-based implementation allows you to create, train, evaluate, and use neural networks for a wide range of applications using a highly user-friendly console/GUI interface or directly using the `NeuralNetwork` class for your own needs if necessary.

## Key Features

- **Realtime Accuracy Graph While Training**: Displays realtime graph of accuracy over epochs for mnist networks while training, allowing users to visualize improvements as they occur.

- **Training Callback Interface**: Provides an interface that includes a callback method that can be passed into the Train function to execute custom code on each mini-batch iteration of training.

- **Customizable Topology**: Define the number of neurons in each layer and activation functions for each layer.
  
- **Multiple Activation Functions**: Supports popular activation functions including linear, sigmoid, tanh, relu, binary, and softmax.

- **Weight Initialization**: Utilizes weight initialization techniques like Xavier (for linear, sigmoid, and tanh) and He (for relu) for better convergence.

- **Training with SGD**: Train your neural network using Stochastic Gradient Descent (SGD) with customizable learning rate, batch size, and decay.

- **Gradient Clipping**: Helps prevent exploding gradients during training by setting a gradient clipping threshold.

- **Regularization Techniques**: Supports popular regularization techniques like L1 and L2 to reduce overfitting by minimizing parameter complexity.

- **Save and Load Models**: Easily save trained models to disk and load them for future use.

## Internal Usage

For use in your own Java projects, simply import the `NeuralNetwork.java` class file and it will immediately be usable. The following section covers the proper syntax for 

1. **Initialize the Neural Network**: Create a neural network by specifying the topology (number of neurons in each layer) and activation functions.

   ```java
   int[] topology = {inputSize, hiddenLayerSize, outputSize};
   String[] activations = {"linear", "relu", "softmax"};
   NeuralNetwork network = new NeuralNetwork(topology, activations);
   network.Init(0.1); //initializes weights and biases according to spread amount
   ```

2. **Training**: Train the neural network using your dataset and desired hyperparameters.

   ```java
   double[][] inputs = // Your input data
   double[][] outputs = // Your output data
   int epochs = 100;
   double learningRate = 0.01;
   int batchSize = 32;
   String lossFunction = "mse"; // or "categorical_crossentropy" for classification
   double decay = 0.001; // Learning rate decay
   network.clipThreshold = 1; //default gradient clipping threshold
   //set regularization of network
   network.SetRegularizationType(NeuralNetwork.RegularizationType.L2); 
   network.SetRegularizationLambda(0.01);
   //train the network with no callback
   network.Train(inputs, outputs, epochs, learningRate, batchSize, lossFunction, decay, null);
   ```

   Or, provide a training callback by passing in a class implementing the static `NeuralNetwork.TrainingCallback` interface as an argument. This can be used to make your own custom train addons like a graph visualization of the data. The `ChartUpdater` class has been provided to visualize accuracy data using this callback interface.

	```java
	public class Callback implements NeuralNetwork.TrainingCallback {
		@Override
		public void onEpochUpdate(int epoch, int batch, double progress, double accuracy) {
			System.out.println("this statement is run for every mini-batch in training");
		}
	}

	class Main {
		public static void main(String[] args) {
			...
			Callback callback = new Callback();
			network.Train(inputs, outputs, epochs, learningRate, batchSize, lossFunction, decay, callback);
		}
	}
	```

3. **Mutation**: Mutate the neural network for a genetic algorithm (evolution).

	```java
	network.Mutate(c, v); //mutates the network with chance c and variation v
	```

5. **Evaluation**: Use the trained model to make predictions and evaluate the cost

	  ```java
	 double[] input = // Your input data
	 double[] prediction = network.Evaluate(input);
		
	 double[] expected = {...};
	 String lossFunction = "mse"; // or "categorical_crossentropy" for classification
	 double cost = network.Cost(prediction, expected, lossFunction);
	  ```

6. **Save and Load**: Save the trained model to disk and load it for future use.

	  ```java
	  // Save the model
	  NeuralNetwork.Save(network, "my_model.nn")
	  
	  // Load the model
	  NeuralNetwork loadedNetwork = NeuralNetwork.Load("my_model.nn");
	  ```

7. **Access/Modify Parameters**: get/set the parameters and information about the network.

	```java
	int numLayers = network.numLayers;
	int[] topology = network.GetTopology(); //topology[i] is # of neurons of layer i
		 
	double[][][] weights = network.GetWeights();
	network.SetWeight(L,n2,n1,w); //sets the weight between layer L neuron n2 and layer L-1 neuron n1 to w

	double[][] biases = network.GetBiases();
	network.SetBias(L,n,b); //sets the bias of layer L neuron n to b

	String activations = network.GetActivations();
	network.SetActivation(L,act); //sets the activation of layer L to act

	double[][] neurons = network.GetNeurons();
	
	String info = network.toString();
	//the following two lines do the same thing
	System.out.println(info);
	System.out.println(network);
	```

## Program Usage

To use this neural network implementation, you can interact with a custom console provided by the program. Follow these steps to get started:

1. **Compile the Code**: First, make sure you are working in the project directory. If you are running the full project with the console interface, run the following commands to compile and run the program:

	***Compile***:

   ```shell
   javac -cp ".:./libraries/jfreechart-1.5.3.jar" Main.java
	```
	***Run***

	```shell
	java -cp ".:./libraries/jfreechart-1.5.3.jar" Main
	```

	Or, if you are just using the `NeuralNetwork` class, the jfreechart library can be excluded, simplifying the commands to:

	***Compile***:

   ```shell
   javac Main.java
	```
	***Run***

	```shell
	java Main
	```

This will launch the program's custom console, allowing you to control and modify neural networks.

#### Available Commands:

- `help`: Display a list of available commands and their descriptions.

- `create`: Create a new neural network by specifying its topology and activation functions.

- `load`: Load a pre-trained neural network from a saved model file.

- `train`: Train the current neural network using your dataset and desired hyperparameters.

- `evaluate`: Use the trained neural network to make predictions on new data.

- `mutate`: Mutate the parameters of the network for a genetic algorithm/implementation

- `mnist`: Initialize/import the MNIST dataset for use in training/evaluating

- `info`: Display information about the neural network's parameters

- `reset`: Reset the current neural network to default

- `modify`: changes parameters of neural network

- `regularization`: changes regularization type (L1, L2, none) and lambda (strength) of neural network

- `magnitude`: Display information about the magnitudes of the network's parameters (min/max/average)

- `cost`: Calculate the cost/accuracy of the network on a test dataset

- `save`: Save the current neural network to a file for future use.

- `exit`: Exit the program.

#### Using the Commands:

- Type a command and press Enter to execute it. Follow the prompts to provide the required information.

- For example, to create a new neural network, you would type `create`, and then follow the prompts to specify the topology and activation functions.

- To train a network, use the `train` command and provide details like the path to the training file or specify `mnist`, number of epochs, learning rate, batch size, loss function, learning decay rate

- For evaluation, use the evaluate command, and input the data you want to predict on, or `mnist [case #]`

- The mnist dataset has been built into the program to allow for building/evaluating hand-drawn digit recognition networks directly using the console and recieving a realtime accuracy visualization to assist with training. mnist networks MUST HAVE input size __784__ and output size __10__. In the majority of cases, the output layer has __softmax__ activation and is trained using the __categorical_crossentropy__ loss function.

#### Save and Load Models:

- You can save the trained model using the save command. This will save the model to a file for future use.

- To load a pre-trained model, use the load command and specify the file path of the saved model.

#### Training/Test Data File Formatting:

All training and test dataset files must be formatted in the following way:

- first line has 3 numbers specifying # cases, input and output size
- every line is a separate training case
- on each line, input is separated by spaces, then equal sign, then output separated by spaces

#### Exiting the Program:

- To exit the program, simply type exit, and the program will terminate.

## Examples

A few neural networks and their training sets have been pre-included into the project, ready to be loaded in. They are:

- `SavedNetwork1`: simple neural network to add 2 numbers
- `SavedNetwork2`: deep neural network to add 2 numbers
- `TrainSet1`: training/test dataset for adding 2 numbers (can be used for `SavedNetwork1` and `SavedNetwork2`)
- `MNISTNetwork`: an untrained neural network with the correct topology to evaluate MNIST cases (digit recognition). accuracy ≈ 12%
- `MNISTNetworkTrained`: a trained neural network that evaluates MNIST cases (digit recognition). accuracy ≈ 87%


