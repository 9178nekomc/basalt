package jsl;

/**
 * Standardized error codes returned across the JNI/native boundary.
 */
public enum JSLExceptionCode {
    OK(0, "Operation succeeded"),
    INVALID_PARAMETER(-1, "Invalid parameter or argument"),
    SHAPE_MISMATCH(-2, "Tensor or matrix shape mismatch"),
    UNKNOWN_HANDLE(-3, "Unknown, stale, or already-released native handle"),
    CONTEXT_CLOSED(-4, "Native context is null or already closed"),
    GRAPH_ERROR(-5, "Computation graph topological or execution error"),
    NATIVE_PANIC(-6, "Native panic caught at the JNI boundary"),
    OUT_OF_MEMORY(-7, "Native memory allocation failed or exceeded limit"),
    UNSUPPORTED_OPERATION(-8, "Unsupported operator or configuration");

    private final int code;
    private final String description;

    JSLExceptionCode(int code, String description) {
        this.code = code;
        this.description = description;
    }

    public int code() {
        return code;
    }

    public String description() {
        return description;
    }

    public static JSLExceptionCode fromCode(int code) {
        for (JSLExceptionCode value : values()) {
            if (value.code == code) {
                return value;
            }
        }
        return INVALID_PARAMETER;
    }
}
