package jsl.concurrent;

import jsl.core.Matrix;
import jsl.ml.Estimator;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Virtual-thread fan-out for batch inference and request orchestration (with automatic fallback
 * to cached daemon threads on JDK 17 runtimes where virtual threads are not present).
 */
public final class InferenceEngine implements AutoCloseable {
    private final ExecutorService executor = createExecutor();
    private final Semaphore permits;
    private volatile boolean closed;

    public InferenceEngine(int parallelism) {
        if (parallelism <= 0) {
            throw new IllegalArgumentException("parallelism must be positive");
        }
        permits = new Semaphore(parallelism);
    }

    private static ExecutorService createExecutor() {
        try {
            Method method = Executors.class.getMethod("newVirtualThreadPerTaskExecutor");
            return (ExecutorService) method.invoke(null);
        } catch (ReflectiveOperationException | SecurityException ignored) {
            return Executors.newCachedThreadPool(runnable -> {
                Thread thread = new Thread(runnable, "jsl-inference-worker");
                thread.setDaemon(true);
                return thread;
            });
        }
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

    /** Runs I/O-bound tasks without a CPU permit. */
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
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException interrupted) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
