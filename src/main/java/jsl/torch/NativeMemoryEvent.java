package jsl.torch;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;

/**
 * JFR event emitted when native context memory snapshots are recorded.
 */
@Name("jsl.NativeMemory")
@Label("JSL Native Memory")
@Category({"JSL", "Observability"})
@Description("Live native tensors, off-heap bytes, and autograd graph node count")
public final class NativeMemoryEvent extends Event {
    @Label("Live Tensors")
    public long liveTensors;

    @Label("Live Native Bytes")
    public long liveBytes;

    @Label("Graph Nodes")
    public long graphNodes;

    @Label("JVM Heap Used Bytes")
    public long heapUsedBytes;
}
