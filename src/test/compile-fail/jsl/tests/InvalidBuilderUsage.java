package jsl.tests;

import jsl.nn.Dense;
import jsl.nn.NeuralNetwork;

/** This file must not compile: image inputs must pass Flatten before Dense. */
final class InvalidBuilderUsage {
    void reject() {
        NeuralNetwork.builder().add(Dense.builder().units(2).build());
    }
}
