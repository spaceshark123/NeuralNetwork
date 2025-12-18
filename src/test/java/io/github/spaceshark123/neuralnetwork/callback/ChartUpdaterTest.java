package io.github.spaceshark123.neuralnetwork.callback;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class ChartUpdaterTest {
	@Test
	public void testJFreeChartAvailability() {
		JFreeChart chart = ChartFactory.createPieChart("Test Chart", null, true, true, false);
		Assertions.assertNotNull(chart);
	}
}