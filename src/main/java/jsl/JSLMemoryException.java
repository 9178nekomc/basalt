package jsl;

/**
 * Thrown when native or off-heap memory allocation fails or exceeds addressable bounds.
 */
public final class JSLMemoryException extends JSLRuntimeException {
    private static final long serialVersionUID = 1L;

    public JSLMemoryException(String message) {
        super(JSLExceptionCode.OUT_OF_MEMORY, message);
    }

    public JSLMemoryException(String message, Throwable cause) {
        super(JSLExceptionCode.OUT_OF_MEMORY, message, cause);
    }
}
