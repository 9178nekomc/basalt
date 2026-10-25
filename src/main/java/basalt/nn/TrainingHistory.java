package basalt.nn;

/** Immutable snapshot of epoch-level training losses. */
public record TrainingHistory(double[] losses) {
    public TrainingHistory {
        losses = losses.clone();
    }

    @Override
    public double[] losses() {
        return losses.clone();
    }

    public int epochs() {
        return losses.length;
    }

    public double lastLoss() {
        if (losses.length == 0) {
            throw new IllegalStateException("No training epochs recorded");
        }
        return losses[losses.length - 1];
    }
}
