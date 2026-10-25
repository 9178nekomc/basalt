package jsl;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import jsl.nativebridge.NativeContext;
import jsl.torch.NativeMemoryEvent;
import jsl.torch.OperatorExecutionEvent;

/**
 * Standardized observability and metrics registry for JSL (Phase 3.2).
 *
 * <p>Tracks operator execution counts, total/mean/p50/p95/p99 latency, throughput
 * (elements/sec and ops/sec), and native/JVM memory footprint, while emitting
 * JFR events ({@link OperatorExecutionEvent}, {@link NativeMemoryEvent}).
 */
public final class JslMetrics {
    private static final int RESERVOIR_SIZE = 256;
    private static final ConcurrentHashMap<String, OperatorTracker> TRACKERS = new ConcurrentHashMap<>();

    private JslMetrics() {}

    /** Snapshot of metrics for a single operator category. */
    public record OperatorSummary(
            String operator,
            long calls,
            long totalElements,
            long totalBytes,
            double totalMillis,
            double meanMicros,
            double p50Micros,
            double p95Micros,
            double p99Micros,
            double maxMicros,
            double throughputElementsPerSec,
            double throughputOpsPerSec) {}

    /** Snapshot of native + JVM memory telemetry. */
    public record MemorySnapshot(
            long liveNativeTensors,
            long liveNativeBytes,
            long graphNodes,
            long jvmHeapUsedBytes,
            long jvmHeapMaxBytes) {}

    /** Full observability snapshot across all operators and memory pools. */
    public record MetricsSnapshot(
            Map<String, OperatorSummary> operators,
            MemorySnapshot memory) {

        public MetricsSnapshot(Map<String, OperatorSummary> operators, MemorySnapshot memory) {
            this.operators = Collections.unmodifiableMap(new LinkedHashMap<>(operators));
            this.memory = memory;
        }

        public String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"memory\": {\n");
            sb.append(String.format(
                    Locale.US,
                    "    \"liveNativeTensors\": %d,\n"
                            + "    \"liveNativeBytes\": %d,\n"
                            + "    \"graphNodes\": %d,\n"
                            + "    \"jvmHeapUsedBytes\": %d,\n"
                            + "    \"jvmHeapMaxBytes\": %d\n"
                            + "  },\n",
                    memory.liveNativeTensors(),
                    memory.liveNativeBytes(),
                    memory.graphNodes(),
                    memory.jvmHeapUsedBytes(),
                    memory.jvmHeapMaxBytes()));
            sb.append("  \"operators\": {\n");
            int idx = 0;
            for (OperatorSummary s : operators.values()) {
                if (idx++ > 0) {
                    sb.append(",\n");
                }
                sb.append(String.format(
                        Locale.US,
                        "    \"%s\": {\"calls\": %d, \"elements\": %d, \"bytes\": %d, "
                                + "\"totalMs\": %.3f, \"meanUs\": %.2f, \"p50Us\": %.2f, "
                                + "\"p95Us\": %.2f, \"p99Us\": %.2f, \"maxUs\": %.2f, "
                                + "\"opsPerSec\": %.1f, \"elementsPerSec\": %.1f}",
                        s.operator(),
                        s.calls(),
                        s.totalElements(),
                        s.totalBytes(),
                        s.totalMillis(),
                        s.meanMicros(),
                        s.p50Micros(),
                        s.p95Micros(),
                        s.p99Micros(),
                        s.maxMicros(),
                        s.throughputOpsPerSec(),
                        s.throughputElementsPerSec()));
            }
            sb.append("\n  }\n}");
            return sb.toString();
        }

        public String toPrometheus() {
            StringBuilder sb = new StringBuilder();
            sb.append("# HELP jsl_native_live_tensors Live native tensors in context\n");
            sb.append("# TYPE jsl_native_live_tensors gauge\n");
            sb.append("jsl_native_live_tensors ").append(memory.liveNativeTensors()).append('\n');
            sb.append("# HELP jsl_native_live_bytes Live native off-heap bytes allocated\n");
            sb.append("# TYPE jsl_native_live_bytes gauge\n");
            sb.append("jsl_native_live_bytes ").append(memory.liveNativeBytes()).append('\n');
            sb.append("# HELP jsl_jvm_heap_used_bytes JVM heap memory used in bytes\n");
            sb.append("# TYPE jsl_jvm_heap_used_bytes gauge\n");
            sb.append("jsl_jvm_heap_used_bytes ").append(memory.jvmHeapUsedBytes()).append('\n');
            for (OperatorSummary s : operators.values()) {
                sb.append(String.format(
                        Locale.US,
                        "jsl_operator_calls_total{op=\"%s\"} %d\n"
                                + "jsl_operator_latency_micros{op=\"%s\",quantile=\"0.5\"} %.2f\n"
                                + "jsl_operator_latency_micros{op=\"%s\",quantile=\"0.95\"} %.2f\n"
                                + "jsl_operator_latency_micros{op=\"%s\",quantile=\"0.99\"} %.2f\n"
                                + "jsl_operator_throughput_elements_per_sec{op=\"%s\"} %.1f\n",
                        s.operator(),
                        s.calls(),
                        s.operator(),
                        s.p50Micros(),
                        s.operator(),
                        s.p95Micros(),
                        s.operator(),
                        s.p99Micros(),
                        s.operator(),
                        s.throughputElementsPerSec()));
            }
            return sb.toString();
        }
    }

    private static final class OperatorTracker {
        private final String name;
        private final LongAdder calls = new LongAdder();
        private final LongAdder totalNanos = new LongAdder();
        private final LongAdder totalElements = new LongAdder();
        private final LongAdder totalBytes = new LongAdder();
        private final AtomicLong maxNanos = new AtomicLong();
        private final long[] reservoir = new long[RESERVOIR_SIZE];
        private final AtomicLong sampleIndex = new AtomicLong();

        OperatorTracker(String name) {
            this.name = name;
        }

        void record(long nanos, long elements, long bytes) {
            calls.increment();
            totalNanos.add(nanos);
            totalElements.add(Math.max(0L, elements));
            totalBytes.add(Math.max(0L, bytes));
            maxNanos.accumulateAndGet(nanos, Math::max);
            long idx = sampleIndex.getAndIncrement();
            reservoir[(int) (idx & (RESERVOIR_SIZE - 1))] = nanos;
        }

        OperatorSummary snapshot() {
            long c = calls.sum();
            long ns = totalNanos.sum();
            long el = totalElements.sum();
            long by = totalBytes.sum();
            long max = maxNanos.get();
            int filled = (int) Math.min(c, (long) RESERVOIR_SIZE);
            double p50 = 0.0;
            double p95 = 0.0;
            double p99 = 0.0;
            if (filled > 0) {
                long[] copy = new long[filled];
                System.arraycopy(reservoir, 0, copy, 0, filled);
                Arrays.sort(copy);
                p50 = copy[Math.min(filled - 1, (int) Math.floor(0.50 * (filled - 1)))] / 1_000.0;
                p95 = copy[Math.min(filled - 1, (int) Math.floor(0.95 * (filled - 1)))] / 1_000.0;
                p99 = copy[Math.min(filled - 1, (int) Math.floor(0.99 * (filled - 1)))] / 1_000.0;
            }
            double totalSec = Math.max(1e-9, ns / 1_000_000_000.0);
            double meanUs = c > 0 ? (ns / 1_000.0) / c : 0.0;
            return new OperatorSummary(
                    name,
                    c,
                    el,
                    by,
                    ns / 1_000_000.0,
                    meanUs,
                    p50,
                    p95,
                    p99,
                    max / 1_000.0,
                    c > 0 ? el / totalSec : 0.0,
                    c > 0 ? c / totalSec : 0.0);
        }
    }

    /** Records an operator execution and emits a JFR {@link OperatorExecutionEvent}. */
    public static void recordOperator(
            String operator, long durationNanos, long elements, long bytesProcessed) {
        if (operator == null || operator.isBlank()) {
            return;
        }
        long safeNanos = Math.max(1L, durationNanos);
        TRACKERS.computeIfAbsent(operator, OperatorTracker::new)
                .record(safeNanos, elements, bytesProcessed);
        OperatorExecutionEvent event = new OperatorExecutionEvent();
        if (event.shouldCommit()) {
            event.operator = operator;
            event.elements = elements;
            event.bytes = bytesProcessed;
            event.durationNanos = safeNanos;
            event.commit();
        }
    }

    /** Overload including backend tag. */
    public static void recordOperator(
            String operator, String backend, long durationNanos, long elements, long bytesProcessed) {
        recordOperator(operator, durationNanos, elements, bytesProcessed);
    }

    /** Times a supplier block, records operator metrics & JFR event, and returns its result. */
    public static <T> T time(String operator, long elements, long bytesProcessed, Supplier<T> block) {
        long start = System.nanoTime();
        try {
            return block.get();
        } finally {
            recordOperator(operator, System.nanoTime() - start, elements, bytesProcessed);
        }
    }

    /** Captures a point-in-time snapshot across all tracked operators and memory pools. */
    public static MetricsSnapshot snapshot(NativeContext ctx) {
        Map<String, OperatorSummary> map = new LinkedHashMap<>();
        for (Map.Entry<String, OperatorTracker> e : TRACKERS.entrySet()) {
            map.put(e.getKey(), e.getValue().snapshot());
        }
        long liveTensors = 0L;
        long liveBytes = 0L;
        long graphNodes = 0L;
        if (ctx != null && ctx.isOpen()) {
            try {
                long[] s = ctx.stats();
                liveTensors = s[0];
                liveBytes = s[1];
                graphNodes = s[2];
            } catch (RuntimeException ignored) {
                liveTensors = 0L;
                liveBytes = 0L;
                graphNodes = 0L;
            }
        }
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        long max = rt.maxMemory();
        NativeMemoryEvent memEvent = new NativeMemoryEvent();
        if (memEvent.shouldCommit()) {
            memEvent.liveTensors = liveTensors;
            memEvent.liveBytes = liveBytes;
            memEvent.graphNodes = graphNodes;
            memEvent.heapUsedBytes = used;
            memEvent.commit();
        }
        MemorySnapshot mem = new MemorySnapshot(liveTensors, liveBytes, graphNodes, used, max);
        return new MetricsSnapshot(map, mem);
    }

    public static MetricsSnapshot snapshot() {
        return snapshot(null);
    }

    /** Resets all operator counters. */
    public static void reset() {
        TRACKERS.clear();
    }
}
