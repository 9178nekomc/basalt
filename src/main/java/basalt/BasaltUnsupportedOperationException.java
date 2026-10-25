package basalt;

/**
 * Thrown when a requested operator, opcode, or model configuration is not supported.
 */
public final class BasaltUnsupportedOperationException extends BasaltRuntimeException {
    private static final long serialVersionUID = 1L;

    public BasaltUnsupportedOperationException(String message) {
        super(BasaltExceptionCode.UNSUPPORTED_OPERATION, message);
    }

    public BasaltUnsupportedOperationException(String message, Throwable cause) {
        super(BasaltExceptionCode.UNSUPPORTED_OPERATION, message, cause);
    }
}
