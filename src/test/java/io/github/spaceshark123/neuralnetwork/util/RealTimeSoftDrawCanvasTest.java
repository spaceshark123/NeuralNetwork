package io.github.spaceshark123.neuralnetwork.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

import javax.swing.JComponent;

public class RealTimeSoftDrawCanvasTest {
	@Test
	public void testSwingAvailability() {
		JComponent component = new JComponent() {};
		Assertions.assertNotNull(component);
	}
}
