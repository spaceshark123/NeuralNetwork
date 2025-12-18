package com.github.spaceshark123.neuralnetwork.callback;


/**
 * Callback interface for monitoring training progress.
 */
public interface TrainingCallback {
	/**
	 * Called after each epoch update during training.
	 * 
	 * @param epoch current epoch number
	 * @param batch current batch number within the epoch
	 * @param progress progress through the epoch (0.0 to 1.0)
	 * @param trainAccuracy training accuracy after this batch
	 * @param testAccuracy test accuracy after this batch (-1 if not available)
	 */
	void onEpochUpdate(int epoch, int batch, double progress, double trainAccuracy, double testAccuracy);
}