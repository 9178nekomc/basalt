package basalt;

/**
 * Thrown when tensor or matrix dimensions are incompatible for an operation.
 */
public final class BasaltShapeException extends BasaltRuntimeException {
    private static final long serialVersionUID = 1L;

    public BasaltShapeException(String message) {
        super(BasaltExceptionCode.SHAPE_MISMATCH, message);
    }

    public BasaltShapeException(String message, Throwable cause) {
        super(BasaltExceptionCode.SHAPE_MISMATCH, message, cause);
    }
}
