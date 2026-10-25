package basalt;

/**
 * Thrown when native or off-heap memory allocation fails or exceeds addressable bounds.
 */
public final class BasaltMemoryException extends BasaltRuntimeException {
    private static final long serialVersionUID = 1L;

    public BasaltMemoryException(String message) {
        super(BasaltExceptionCode.OUT_OF_MEMORY, message);
    }

    public BasaltMemoryException(String message, Throwable cause) {
        super(BasaltExceptionCode.OUT_OF_MEMORY, message, cause);
    }
}
