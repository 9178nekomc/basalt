package jsl.torch;

import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;

/**
 * JFR event emitted for every native command-queue execution.
 *
 * <p>JFR is a sampling, production-grade profiler built into the JDK. Python's standard profiler
 * is invasive; this event stream is always available and can be enabled at runtime with
 * {@code jcmd <pid> JFR.start}. It records op count, thread hint, bytes and wall duration without
 * a Python interpreter in the hot path.</p>
 */
@Name("jsl.NativeQueue")
@Label("Native Command Queue")
@Category({"JSL", "Native"})
@Description("One batched JNI execution of the Rust tensor engine")
public final class NativeQueueEvent extends Event {
    @Label("Commands")
    public int commands;

    @Label("Threads")
    public int threads;

    @Label("Bytes")
    public int bytes;

    @Label("Duration (ns)")
    public long durationNanos;
}
