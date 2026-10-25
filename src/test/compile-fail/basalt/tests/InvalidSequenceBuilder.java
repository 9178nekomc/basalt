package basalt.tests;

import basalt.nn.Dense;
import basalt.nn.SequenceNetwork;

/** This file must not compile: a Dense layer is BC -&gt; BC, not BSF -&gt; BSF. */
final class InvalidSequenceBuilder {
    void reject() {
        SequenceNetwork.builder().add(Dense.builder().units(2).build());
    }
}
