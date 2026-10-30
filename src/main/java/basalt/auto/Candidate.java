package basalt.auto;

/**
 * One row of the screening table produced by {@link Auto}.
 *
 * @param algorithm registry name of the screened candidate
 * @param score     mean cross-validated score; higher is always better, so a lower-is-better
 *                  metric such as RMSE is negated before it is stored here
 * @param selected  whether this candidate won the screening and was refitted on the full data
 */
public record Candidate(String algorithm, double score, boolean selected) {

    @Override
    public String toString() {
        return String.format("%-28s %8.4f%s", algorithm, score, selected ? "  <- selected" : "");
    }
}
