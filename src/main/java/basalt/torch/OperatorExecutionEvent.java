package basalt.torch;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;

/**
 * JFR event emitted for individual tensor operators, classical ML fits, and batch inference calls.
 */
@Name("basalt.OperatorExecution")
@Label("Basalt Operator Execution")
@Category({"Basalt", "Observability"})
@Description("Execution timing, memory footprint, and sample throughput of a Basalt operator or estimator")
public final class OperatorExecutionEvent extends Event {
    @Label("Operator")
    public String operator;

    @Label("Samples / Elements")
    public long elements;

    @Label("Bytes Processed")
    public long bytes;

    @Label("Duration (ns)")
    public long durationNanos;
}
