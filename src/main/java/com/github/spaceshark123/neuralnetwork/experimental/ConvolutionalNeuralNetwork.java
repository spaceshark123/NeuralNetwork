package com.github.spaceshark123.neuralnetwork.experimental;

import java.util.ArrayList;
import com.github.spaceshark123.neuralnetwork.NeuralNetwork;
import com.github.spaceshark123.neuralnetwork.activation.ActivationFunction;

/**
 * Convolutional Neural Network (CNN) implementation.
 * Supports convolutional, pooling, and dense layers.
 * 
 * NOT FULLY IMPLEMENTED YET
 * DO NOT USE THIS CLASS
 */
public class ConvolutionalNeuralNetwork extends NeuralNetwork {
    protected String[] layerTypes; // "convolution", "pooling", "dense"
    protected ArrayList<double[][][]>[] filters; // filters[layer].get(filter #)[x][y][channel]
    protected int[] filterSizes; // filterSizes[layer] (square filter) of size filterSizes[layer] x filterSizes[layer]

    protected int inputWidth;
    protected int inputHeight;
    protected int[] numChannels; // number of channels in the input image, number of filters for convolution layers

    //topology only matters for dense layers, filtercounts only matters for convolution layers
    public ConvolutionalNeuralNetwork(int inputWidth, int inputHeight, int[] numChannels, int[] topology, ActivationFunction[] active,
            String[] layerTypes, int[] filterSizes, int[] filterCounts) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.numChannels = numChannels;

        //preprocess topology to calculate number of neurons per layer given convolutional and pooling layers
        topology[0] = inputWidth * inputHeight * numChannels[0];
        for (int i = 1; i < topology.length; i++) {
            if (layerTypes[i].equals("convolution")) {
                //same padding
                topology[i] = topology[i - 1];
            } else if (layerTypes[i].equals("pooling")) {
                //valid padding
                topology[i] = topology[i - 1] / (filterSizes[i] * filterSizes[i]);
            }
        }

        int maxLayerSize = max(topology);
        neuronsPerLayer = topology.clone();
        numLayers = topology.length;
        neurons = new double[numLayers][maxLayerSize];
        neuronsRaw = new double[numLayers][maxLayerSize];
        biases = new double[numLayers][maxLayerSize];
        weights = new double[numLayers][maxLayerSize][maxLayerSize]; //weights only matter for dense layers
        activations = active.clone();

        int maxFilterCount = max(filterCounts);

        this.layerTypes = layerTypes;
        this.filterSizes = filterSizes;
        filters = new ArrayList[numLayers];
        for (int i = 1; i < numLayers; i++) {
            if (layerTypes[i].equals("convolution")) {
                filters[i] = new ArrayList(numChannels[i]);
                for (int j = 0; j < numChannels[i]; j++) {
                    filters[i].add(new double[filterSizes[i]][filterSizes[i]][numChannels[i - 1]]);
                }
            } else if (layerTypes[i].equals("pooling")) {
                filters[i] = new ArrayList(numChannels[i]);
                for (int j = 0; j < numChannels[i]; j++) {
                    filters[i].add(new double[filterSizes[i]][filterSizes[i]][numChannels[i - 1]]);
                }
            }
        }
    }

    public ConvolutionalNeuralNetwork(int inputWidth, int inputHeight, int[] numChannels, int[] topology, ActivationFunction[] active,
    String[] layerTypes, int[] filterSizes, int[] filterCounts, RegularizationType regularizationType,
			double regularizationStrength) {
		this(inputWidth, inputHeight, numChannels, topology, active, layerTypes, filterSizes, filterCounts);
		//set regularization
		this.regularizationType = regularizationType;
		lambda = regularizationStrength;
	}

    public ConvolutionalNeuralNetwork() {

    }
    
    //initialize network with random starting values
	public void init(double biasSpread) {
		clearNeurons();
		initWeights();
		initBiases(biasSpread);
	}
    
    //function to convert a 2d array index to a 1d array index
    public int index(int x, int y, int width) {
        return y * width + x;
    }

    //function to convert a 3d array index to a 1d array index
    public int index(int x, int y, int z, int width, int height) {
        return z * width * height + y * width + x;
    }
}
