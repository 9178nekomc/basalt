package basalt.torch;

import basalt.core.Layout;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Group Normalization over {@code [N, C, ...]} tensors where {@code C} channels are partitioned
 * into {@code numGroups} groups, independent of batch size, matching {@code torch.nn.GroupNorm}.
 */
public final class GroupNorm<L extends Layout> extends Module<L, L> {
    private final int numGroups;
    private final int numChannels;
    private final float epsilon;
    private final Tensor gamma;
    private final Tensor beta;

    public GroupNorm(NativeContext context, int numGroups, int numChannels) {
        this(context, numGroups, numChannels, 1e-5f);
    }

    public GroupNorm(NativeContext context, int numGroups, int numChannels, float epsilon) {
        if (numGroups <= 0 || numChannels <= 0 || numChannels % numGroups != 0) {
            throw new IllegalArgumentException(
                    "numChannels (" + numChannels + ") must be positive and divisible by numGroups ("
                            + numGroups + ")");
        }
        this.numGroups = numGroups;
        this.numChannels = numChannels;
        this.epsilon = epsilon;
        this.gamma = Tensor.parameter(context, numChannels);
        this.beta = Tensor.parameter(context, numChannels);
        this.gamma.fill(1.0f);
        this.beta.fill(0.0f);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.groupNorm(gamma, beta, numGroups, epsilon);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>(2);
        params.add(gamma);
        params.add(beta);
        return params;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        return List.of(Map.entry("gamma", gamma), Map.entry("beta", beta));
    }

    public int numGroups() {
        return numGroups;
    }

    public int numChannels() {
        return numChannels;
    }

    public Tensor gamma() {
        return gamma;
    }

    public Tensor beta() {
        return beta;
    }
}
