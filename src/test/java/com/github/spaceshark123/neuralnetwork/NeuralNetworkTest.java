package com.github.spaceshark123.neuralnetwork;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;

public class NeuralNetworkTest {

	@Test
	public void testAddition() {
		int sum = 2 + 3;
		Assertions.assertEquals(5, sum);
	}

	@Test
	public void testJFreeChartAvailability() {
		JFreeChart chart = ChartFactory.createPieChart("Test Chart", null, true, true, false);
		Assertions.assertNotNull(chart);
	}
}
