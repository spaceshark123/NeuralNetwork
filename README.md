
# Neural Network in Java

This project provides a flexible and extensible implementation of a multithreaded feedforward neural network in Java with popular optimizers included, wrapped up in a console user interface with a realtime accuracy graph visualizer while training. The neural network is designed to be easy to use, import and customize, making it a valuable tool for various machine learning and deep learning tasks. This was made without any external machine learning, math, or other libraries to aid in its creation (pure Java, at least for the base class). Maven is used for dependency management, along with a Maven wrapper for ease of use.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Internal Usage](#internal-usage)
- [Program Usage](#program-usage)
- [Examples](#examples)

## Overview

Neural networks are a fundamental building block of modern machine learning and artificial intelligence. This Java-based implementation allows you to create, train, evaluate, and use neural networks for a wide range of applications using a highly user-friendly console/GUI interface or directly using the `NeuralNetwork` class in the `com.github.spaceshark123.neuralnetwork` package for your own needs if necessary.

## Key Features

- **User-friendly Console Interface**: Everything is wrapped in an easy-to-use console interface using commands to perform tasks, similar to a shell terminal. This allows for no-code experimentation with neural networks.

- **Realtime Accuracy Graph While Training**: Displays realtime graph of accuracy over epochs for mnist networks while training, allowing users to visualize improvements as they occur.

- **Realtime MNIST Drawing Tool**: Includes a drawing tool that allows users to draw on a 28x28 canvas and recieve realtime predictions by the network of which MNIST digit it is. This allows for easy debugging and evaluation.

- **Parallelism and Multithreading**: Uses parallel computing and multithreading to dramatically accelerate computation time for evaluation and training mini-batches.

- **Training Callback Interface**: Provides an interface that includes a callback method that can be passed into the Train function to execute custom code on each mini-batch iteration of training.

- **Customizable Topology**: Define the number of neurons in each layer and activation functions for each layer.
  
- **Multiple Activation Functions**: Supports popular activation functions including linear, sigmoid, tanh, relu, leaky relu, binary, and softmax.

- **Multiple Loss Functions**:  Supports major loss/error functions: Mean Squared Error, Sum Squared Error, and Categorical Cross-Entropy.

- **Multiple Optimizers**: Includes popular optimizers like Stochastic gradient descent (SGD), SGD with momentum, AdaGrad, RMSProp, and Adam. Supports additional, custom optimizers using the `Optimizer` interface from the `com.github.spaceshark123.neuralnetwork.optimizer` package.

- **Weight Initialization**: Utilizes weight initialization techniques like Xavier (for linear, sigmoid, and tanh) and He (for relu) for better convergence.

- **Data Augmentation**: Includes option to augment MNIST dataset on import, which performs random affine transformations (translation, rotation, scaling) to increase generalizability of networks.

- **Training**: Train your neural network using mini-batched Gradient Descent (SGD) with customizable optimizer, loss function, learning rate, mini-batch size, and learning rate decay.

- **Train/Test Set Cross-Validation**: During training, train and test set accuracy are measured for each mini-batch, ensuring a complete picture of network performance and prevention of overfitting. The console interface allows users to specify a train/test split ratio.

- **Gradient Clipping**: Helps prevent exploding gradients during training by setting a gradient clipping threshold.

- **Regularization Techniques**: Supports popular regularization techniques like L1 and L2 to reduce overfitting by minimizing parameter complexity.

- **Save and Load Models**: Easily save trained models to disk and load them for future use.

## Internal Usage

The following packages are available for use:

- `com.github.spaceshark123.neuralnetwork`: Contains the main `NeuralNetwork` class and related interfaces for optimizers and training callbacks.
- `com.github.spaceshark123.neuralnetwork.cli`: Contains the console interface for user interaction with the neural network. Intended for direct program running rather than import.
- `com.github.spaceshark123.neuralnetwork.optimizer`: Contains optimizer implementations and the `Optimizer` interface for creating custom optimizers.
- `com.github.spaceshark123.neuralnetwork.activation`: Contains the `ActivationFunction` interface and various activation function implementations (Linear, Sigmoid, Tanh, ReLU, LeakyReLU, Binary, Softmax) for use in the neural network.
- `com.github.spaceshark123.neuralnetwork.callback`: Contains the `TrainingCallback` interface for creating custom training callbacks and the `ChartUpdater` class for visualizing accuracy data in realtime during training.
- `com.github.spaceshark123.neuralnetwork.experimental`: Contains experimental and not fully implemented classes like `ConvolutionalNeuralNetwork`. DO NOT USE THESE CLASSES.

For use in your own Java projects, simply import the relevant packages/classes and it will immediately be usable. The following section covers the proper syntax for:

1. **Initialize the Neural Network**: Create a neural network by specifying the topology (number of neurons in each layer) and activation functions (from the `com.github.spaceshark123.neuralnetwork.activation` package).

   ```java
   int[] topology = {inputSize, hiddenLayerSize, outputSize};
   ActivationFunction[] activations = {
      new ReLU(), // example of relu activation
      new LeakyReLU(0.01), // example of leaky relu with alpha = 0.01
      new Softmax() // example of softmax activation
   };
   NeuralNetwork network = new NeuralNetwork(topology, activations);
   network.init(0.1); //initializes weights and biases according to spread amount
   ```

2. **Training**: Train the neural network using your train and test/validation datasets and desired hyperparameters, taking advantage of multiple CPU cores to speed up training time.

   ```java
   double[][] trainInputs = {...};
   double[][] trainOutputs = {...};

   double[][] testInputs = {...};
   double[][] testOutputs = {...};
   int epochs = 100;
   double learningRate = 0.01;
   int batchSize = 32;
   String lossFunction = "mse"; // or "sse" or "categorical_crossentropy"
   double decay = 0.1; // Learning rate decay
   double momentum = 0.9;
   network.clipThreshold = 1; //default gradient clipping threshold
   //set regularization of network
   network.setRegularizationType(NeuralNetwork.RegularizationType.L2); 
   network.setRegularizationLambda(0.001);
   Optimizer optimizer = new Adam(0.9, 0.999); //specify optimizer for training
   //train the network with no callback
   network.train(trainInputs, trainOutputs, testInputs, testOutputs, epochs, learningRate, batchSize, lossFunction, decay, optimizer, null);
   ```

optimizers implement the `Optimizer` interface and included optimizers are found in the `com.github.spaceshark123.neuralnetwork.optimizer` package. Included optimizers are:

- `SGD()`
- `SGDMomentum(double momentum)`
- `AdaGrad()`
- `RMSProp(double decayRate)`
- `Adam(double beta1, double beta2)`

Custom optimizers can be made by creating a class implementing the `Optimizer` interface from the `com.github.spaceshark123.neuralnetwork.optimizer` package. This can be used to create other optimizers not already included in the NeuralNetwork class.

 ```java
 public static class CustomOptimizer implements Optimizer {
  //assign to the elements of biases and weights in the step function
  private double[][] biases;
  private double[][][] weights;
  private int[] neuronsPerLayer;

  @Override
  public void initialize(int[] neuronsPerLayer, double[][] biases, double[][][] weights) {
   this.biases = biases;
   this.weights = weights;
   this.neuronsPerLayer = neuronsPerLayer;
   //other initializations
   ...
  }

  @Override
  public void step(double[][] avgBiasGradient, double[][][] avgWeightGradient, double learningRate) {
   for (int i = 1; i < neuronsPerLayer.length; i++) {
    for (int j = 0; j < neuronsPerLayer[i]; j++) {
     //set biases
     biases[i][j] = ...
     for (int k = 0; k < neuronsPerLayer[i - 1]; k++) {
      //set weights
      weights[i][j][k] = ...
     }
    }
   }
  }
 }
 ```

Activation functions implement the `ActivationFunction` interface and are found in the `com.github.spaceshark123.neuralnetwork.activation` package. Included activation functions are:

- `Linear()`
- `Sigmoid()`
- `Tanh()`
- `ReLU()`
- `LeakyReLU(double alpha)`
- `Binary()`
- `Softmax()`

Custom activation functions can be created by implementing the `ActivationFunction` interface and registering them as a valid activation function with the `ActivationFunctionFactory` using its `register` method. The following example shows how to create a custom activation function with one parameter `param1` and register it with the factory.

```java
 public class CustomActivation implements ActivationFunction {
private final double param1;

 public CustomActivation(double param1) {
   this.param1 = param1;
 }

  @Override
  public double activate(double x) {
   //custom activation code
   ...
  }

  @Override
  public double derivative(double x) {
   //custom derivative code
   ...
  }

  @Override
  public String getName() {
   return "CUSTOM"; // display name of activation function
 }

 @Override
 public String toConfigString() {
  return "CUSTOM(param1=" + param1 + ")"; // string used in config/saved files to uniquely identify this activation function
 }
}

 //register the custom activation function with the factory
 ActivationFunctionFactory.register("CUSTOM", params -> {
    double param1 = params.containsKey("param1")
        ? Double.parseDouble(params.get("param1")) 
        : 1.0;
    return new CustomActivation(param1);
});
```

Optionally, provide a custom training callback by passing in a class implementing the `TrainingCallback` interface from the `com.github.spaceshark123.neuralnetwork.callback` package as an argument. This can be used to make your own custom train addons like a graph visualization of the data. The `ChartUpdater` class has been provided in the same package to visualize accuracy data using this callback interface.

 ```java
 public class Callback implements TrainingCallback {
  //testAccuracy is -1 if the current mini-batch doesn't have a test accuracy
  @Override
  public void onEpochUpdate(int epoch, int batch, double progress, double trainAccuracy, double testAccuracy) {
   System.out.println("this statement is run for every mini-batch in training");
  }
 }

 class Main {
  public static void main(String[] args) {
   ...
   Callback callback = new Callback();
   network.train(inputs, outputs, epochs, learningRate, batchSize, lossFunction, decay, optimizer, callback);
  }
 }
 ```

3. **Mutation**: Mutate the neural network for a genetic algorithm (evolution).

 ```java
 network.mutate(c, v); //mutates the network with chance c and variation v
 ```

5. **Evaluation**: Use the trained model to make predictions and evaluate the loss

  ```java
  double[] input = {...};
  double[] prediction = network.evaluate(input);
  
  double[] expected = {...};
  String lossFunction = "mse"; // or "sse" or "categorical_crossentropy"
  double loss = network.loss(prediction, expected, lossFunction);
  ```

6. **Save and Load**: Save the trained model to disk and load it for future use, either as a java object, which isn't human readable and doesn't transfer between programming languages but is faster, or a plain text file containing parameters, which is human readable and also transferrable between programming languages.

  ```java
  // Save the model as a java object
  NeuralNetwork.save(network, "my_model_java.nn");
  // Load the model from a file formatted as a java object
  NeuralNetwork loadedNetwork = NeuralNetwork.load("my_model_java.nn");

  // Save the model as a plain text file
  NeuralNetwork.saveParameters(network, "my_model.txt");
  // Load the model from a plain text file
  NeuralNetwork loadedTxtNetwork = NeuralNetwork.load("my_model.txt");
  ```

   The plain text file is separated into lines that each contain a unique set of parameters specified by the first token and followed by the corresponding values, all separated by spaces. For example, one line could contain: `topology 784 512 10`, which would translate to a neural network with 3 layers of those sizes. The headings and their specifications are as follows:

- `numlayers`: contains an integer for the number of layers in the network. usually the first line.
- `topology`: contains `numlayers` integers describing the number of neurons in each layer. usually the second line.
- `activations`: contains `numlayers` strings describing the activation functions for each layer. This includes the input layer, even though it is never used. The valid activation functions are contained in the registered activation functions of the `ActivationFunctionFactory` class. By default, these are: LINEAR, SIGMOID, TANH, RELU, LEAKYRELU(param1={value}), BINARY, SOFTMAX.
- `regularization`: contains an all-caps string describing the mode of regularization and a decimal for the lambda value (regularization strength)
- `biases`: contains all the biases for all the layers. in order from input to output layer, first neuron to last neuron for each layer. includes the input layer, even though it is never used.
- `weights`: contains all the weights. Internally, weights is represented as a 3D array with 1st dimension layer, 2nd dimension neuron #, and 3rd dimension incoming neuron # from previous layer. All weights are flattened into series in order from input to output layer, first neuron to last neuron for each layer, and first neuron to last neuron for each previous layer.

7. **Access/Modify Parameters**: get/set the parameters and information about the network.

 ```java
 int numLayers = network.numLayers;
 int[] topology = network.getTopology(); //topology[i] is # of neurons of layer i
   
 double[][][] weights = network.getWeights();
 network.setWeight(L,n2,n1,w); //sets the weight between layer L neuron n2 and layer L-1 neuron n1 to w

 double[][] biases = network.getBiases();
 network.setBias(L,n,b); //sets the bias of layer L neuron n to b

 ActivationFunction[] activations = network.getActivations();
 network.setActivation(L,new ReLU()); //sets the activation of layer L to relu

 double[][] neurons = network.getNeurons();
 
 String info = network.toString();
 //the following two lines do the same thing
 System.out.println(info);
 System.out.println(network);
 ```

## Program Usage

To use this neural network implementation, you can interact with a custom console provided by the program. A maven wrapper has been included for use. If you do not have Maven installed (check with `mvn -v`), you can use the wrapper scripts to build and run the project by replacing `mvn` with `./mvnw` (Linux/Mac) or `mvnw.cmd` (Windows) in the commands below:

1. **Build the JAR**: First, make sure you are working in the project directory. If you are running the full project with the console interface, run the following commands to compile and run the program:

   **Compile**:

   ```shell
   mvn package
   ```

   **Run**:

   ```shell
   # Run the produced JAR (replace <version> with the actual version in target/)
   java -jar target/NeuralNetwork-<version>.jar

   # Or use a glob to avoid typing the version (works if only one matching JAR exists)
   java -jar target/NeuralNetwork-*.jar
   ```

This will launch the program's custom console, allowing you to control and modify neural networks.

Or, if you are using the library, you can compile and run your own Java files that import the `NeuralNetwork` class directly using the `import com.github.spaceshark123.neuralnetwork.NeuralNetwork;` statement.

#### Available Commands

- `help`: Display a list of available commands and their descriptions.

- `create`: Create a new neural network by specifying its topology and activation functions.

- `init`: initializes the neural network parameters (weights and biases) with the specified bias spread and weight initialization method. ('he' or 'xavier')

- `load`: Load a pre-trained neural network from a saved model file.

- `train`: Train the current neural network using your dataset, loss function, optimizer, and desired hyperparameters.

- `evaluate`: Use the trained neural network to make predictions on new data.

- `mutate`: Mutate the parameters of the network for a genetic algorithm/implementation

- `mnist`: Initialize/import the MNIST dataset for use in training/evaluating or draw your own handwritten digits for the network to evaluate in realtime.

- `info`: Display information about the neural network's parameters

- `reset`: Reset the current neural network to default

- `modify`: changes parameters of neural network

- `regularization`: changes regularization type (L1, L2, none) and lambda (strength) of neural network

- `magnitude`: Display information about the magnitudes of the network's parameters (min/max/average)

- `loss`: Calculate the loss/accuracy of the network on a test dataset

- `save`: Save the current neural network to a file for future use.

- `exit`: Exit the program.

#### Using the Commands

- Type a command and press Enter to execute it. Follow the prompts to provide the required information.

- For example, to create a new neural network, you would type `create`, and then follow the prompts to specify the topology and activation functions.

- To train a network, use the `train` command and provide details like the path to the training file or specify `mnist`, number of epochs, learning rate, mini-batch size, loss function, optimizer, learning decay rate

- For evaluation, use the evaluate command, and input the data you want to predict on, or `mnist [case #]`

- The mnist dataset has been built into the program to allow for building/evaluating hand-drawn digit recognition networks directly using the console and recieving a realtime accuracy visualization to assist with training. You can run `mnist test` to draw your own hand-written digits for the network to evaluate in realtime. the `mnist` command to import the MNIST dataset has an optional `augmented` modifier to randomly apply affine transformations to the data, making the network more generalizable at the cost of some accuracy and convergence  speed. mnist networks MUST HAVE input size **784** and output size **10**. In the majority of cases, the output layer has **softmax** activation and is trained using the **categorical_crossentropy** loss function.

#### Save and Load Models

- You can save the trained model using the save command. This will save the model to a file for future use.

- To load a pre-trained model, use the load command and specify the file path of the saved model.

#### Training/Test Data File Formatting

All training and test dataset files must be formatted in the following way:

- first line has 3 numbers specifying # cases, input and output size
- every line is a separate training case
- on each line, input is separated by spaces, then equal sign, then output separated by spaces

#### Exiting the Program

- To exit the program, simply type exit, and the program will terminate.

## Examples

A few neural networks and their training sets have been pre-included into the project in the `models` and `datasets` directories respectively, ready to be loaded in:

#### `models`

- `MNISTNetwork`: an untrained neural network with the correct topology to evaluate MNIST cases (digit recognition). accuracy ≈ 10% (object mode)
- `MNISTNetworkTrained`: a trained neural network that evaluates MNIST cases (digit recognition) with high, generalized accuracy from training on an augmented data set. Good for testing your own digits. accuracy ≈ 98.14% (parameters mode)

#### `datasets`

- `TrainSet1`: training/test dataset for adding 2 numbers (2 inputs, 1 output) (plain text mode)
- `MNIST`: the MNIST dataset for training/evaluating handwritten digits (stored as `mnist_train.txt` and `mnist_test.txt` in `data` directory), 784 inputs, 10 outputs
- `testTrainSet`: training/test dataset with a single datapoint for testing purposes (plain text mode)
