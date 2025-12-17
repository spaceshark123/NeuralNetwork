package com.github.spaceshark123.neuralnetwork.callback;

public interface TrainingCallback {
	// test accuracy is assigned -1 if not available for the current mini-batch
	void onEpochUpdate(int epoch, int batch, double progress, double trainAccuracy, double testAccuracy);
}