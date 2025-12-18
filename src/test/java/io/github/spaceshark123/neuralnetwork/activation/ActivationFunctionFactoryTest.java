package io.github.spaceshark123.neuralnetwork.activation;

import org.junit.jupiter.api.Test;

import io.github.spaceshark123.neuralnetwork.activation.ActivationFunction;
import io.github.spaceshark123.neuralnetwork.activation.ActivationFunctionFactory;

import org.junit.jupiter.api.Assertions;

public class ActivationFunctionFactoryTest {
	@Test
	public void testGetActivationFunctionByName() {
		ActivationFunction relu = ActivationFunctionFactory.create("ReLU");
		Assertions.assertNotNull(relu);
		Assertions.assertEquals("ReLU", relu.getName());

		ActivationFunction linear = ActivationFunctionFactory.create("Linear");
		Assertions.assertNotNull(linear);
		Assertions.assertEquals("Linear", linear.getName());

		ActivationFunction sigmoid = ActivationFunctionFactory.create("Sigmoid");
		Assertions.assertNotNull(sigmoid);
		Assertions.assertEquals("Sigmoid", sigmoid.getName());

		ActivationFunction softmax = ActivationFunctionFactory.create("Softmax");
		Assertions.assertNotNull(softmax);
		Assertions.assertEquals("Softmax", softmax.getName());
	}

	@Test
	public void testGetActivationFunctionByInvalidName() {
		Assertions.assertThrows(IllegalArgumentException.class, () -> {
			ActivationFunctionFactory.create("InvalidName");
		});
	}

	@Test
	public void testRegister() {
		ActivationFunctionFactory.register("CustomActivation", params -> (new ActivationFunction() {
			private static final long serialVersionUID = 1L;

			@Override
			public double activate(double raw) {
				return raw; // Identity
			}

			@Override
			public double derivative(double raw) {
				return 1.0;
			}

			@Override
			public String getName() {
				return "CustomActivation";
			}

			@Override
			public String toConfigString() {
				return "CustomActivation";
			}
		}));

		ActivationFunction custom = ActivationFunctionFactory.create("CustomActivation");
		Assertions.assertNotNull(custom);
		Assertions.assertEquals("CustomActivation", custom.getName());
		Assertions.assertEquals("CustomActivation", custom.toConfigString());
		Assertions.assertEquals(5.0, custom.activate(5.0));
		Assertions.assertEquals(1.0, custom.derivative(5.0));
		Assertions.assertDoesNotThrow(() -> {
			ActivationFunctionFactory.create("CustomActivation");
		});
		// Clean up by unregistering
		ActivationFunctionFactory.unregister("CustomActivation");
		Assertions.assertThrows(IllegalArgumentException.class, () -> {
			ActivationFunctionFactory.create("CustomActivation");
		});
	}
}
