package jsl.concurrent;

import jsl.core.Matrix;
import jsl.ml.Estimator;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;

/**
 * Virtual-thread fan-out for batch inference and request orchestration.
 *
 * <p>Virtual threads are cheap enough to create one per request, but CPU-bound work still has to
 * be bounded by the number of cores; {@link #predict} therefore gates native/Java kernels with a
 * semaphore. I/O-heavy work uses {@link #runIo} without a gate, where virtual threads win by
 * avoiding one platform thread per request.</p>
 */
public final class InferenceEngine implements AutoCloseable {
    private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
    private final Semaphore permits;
    private boolean closed;

    public InferenceEngine(int parallelism) {
        if (parallelism <= 0) {
            throw new IllegalArgumentException("parallelism must be positive");
        }
        permits = new Semaphore(parallelism);
    }

    /** Predicts independent batches concurrently, bounded by the configured parallelism. */
    public List<double[]> predict(Estimator model, List<Matrix> batches) {
        List<Callable<double[]>> tasks = new ArrayList<>(batches.size());
        for (Matrix batch : batches) {
            tasks.add(() -> {
                permits.acquire();
                try {
                    return model.predict(batch);
                } finally {
                    permits.release();
                }
            });
        }
        return runAll(tasks);
    }

    /** Runs I/O-bound tasks without a CPU permit; one virtual thread per task. */
    public <T> List<T> runIo(List<Callable<T>> tasks) {
        return runAll(tasks);
    }

    public <T> List<T> runAll(List<Callable<T>> tasks) {
        if (closed) {
            throw new IllegalStateException("InferenceEngine is closed");
        }
        List<Future<T>> futures = new ArrayList<>(tasks.size());
        for (Callable<T> task : tasks) {
            futures.add(executor.submit(task));
        }
        List<T> results = new ArrayList<>(tasks.size());
        for (Future<T> future : futures) {
            try {
                results.add(future.get());
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while waiting for inference", failure);
            } catch (ExecutionException failure) {
                throw new IllegalStateException("Inference task failed", failure.getCause());
            }
        }
        return results;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            executor.close();
        }
    }
}
