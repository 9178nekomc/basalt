package basalt.zoo;

import java.util.List;

/**
 * Per-epoch loss record produced by {@link Trainer}.
 *
 * <p>Losses are the mean over the batches of one epoch, so they are comparable across epochs even
 * when the last batch is short.</p>
 */
public final class History {

    private final List<Double> losses;

    History(List<Double> losses) {
        this.losses = List.copyOf(losses);
    }

    /** One entry per completed epoch, oldest first. */
    public List<Double> losses() {
        return losses;
    }

    /** Number of epochs that actually ran. */
    public int epochs() {
        return losses.size();
    }

    /** Loss of the last epoch, or {@code NaN} when nothing ran. */
    public double lastLoss() {
        return losses.isEmpty() ? Double.NaN : losses.get(losses.size() - 1);
    }

    /** Lowest loss seen, or {@code NaN} when nothing ran. */
    public double bestLoss() {
        double best = Double.NaN;
        for (double loss : losses) {
            if (Double.isNaN(best) || loss < best) {
                best = loss;
            }
        }
        return best;
    }

    /** Zero-based index of the epoch that produced {@link #bestLoss()}, or {@code -1}. */
    public int bestEpoch() {
        int bestIndex = -1;
        double best = Double.NaN;
        for (int index = 0; index < losses.size(); index++) {
            double loss = losses.get(index);
            if (Double.isNaN(best) || loss < best) {
                best = loss;
                bestIndex = index;
            }
        }
        return bestIndex;
    }

    @Override
    public String toString() {
        return "History[epochs=" + losses.size() + ", last=" + String.format("%.6f", lastLoss())
                + ", best=" + String.format("%.6f", bestLoss()) + "]";
    }
}
