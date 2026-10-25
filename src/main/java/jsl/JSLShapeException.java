package jsl;

/**
 * Thrown when tensor or matrix dimensions are incompatible for an operation.
 */
public final class JSLShapeException extends JSLRuntimeException {
    private static final long serialVersionUID = 1L;

    public JSLShapeException(String message) {
        super(JSLExceptionCode.SHAPE_MISMATCH, message);
    }

    public JSLShapeException(String message, Throwable cause) {
        super(JSLExceptionCode.SHAPE_MISMATCH, message, cause);
    }
}
