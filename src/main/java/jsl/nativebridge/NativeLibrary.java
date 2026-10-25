package jsl.nativebridge;

import jsl.core.NativeKernels;

/**
 * Availability facade for the Rust tensor runtime.
 *
 * <p>The shared library is loaded once by {@link NativeKernels}; this class only exposes the
 * engine version and a fail-fast guard. Keeping a single loader avoids the "already loaded in
 * another class loader" failure mode when both the classical kernels and the tensor engine are
 * used in the same JVM.</p>
 */
public final class NativeLibrary {
    private NativeLibrary() {
    }

    public static boolean available() {
        return NativeKernels.available();
    }

    public static String loadError() {
        return NativeKernels.loadError();
    }

    public static String libraryPath() {
        return NativeKernels.libraryPath();
    }

    /** Engine ABI version; {@code 0} when the library is unavailable. */
    public static int version() {
        return available() ? NativeBindings.version() : 0;
    }

    public static void requireAvailable() {
        if (!available()) {
            throw new jsl.JSLRuntimeException(
                    "JSL native tensor runtime is unavailable: " + loadError()
                            + " (expected " + libraryPath() + "; run ./native-build.sh)");
        }
    }
}
