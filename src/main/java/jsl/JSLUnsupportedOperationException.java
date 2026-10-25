package jsl;

/**
 * Thrown when a requested operator, opcode, or model configuration is not supported.
 */
public final class JSLUnsupportedOperationException extends JSLRuntimeException {
    private static final long serialVersionUID = 1L;

    public JSLUnsupportedOperationException(String message) {
        super(JSLExceptionCode.UNSUPPORTED_OPERATION, message);
    }

    public JSLUnsupportedOperationException(String message, Throwable cause) {
        super(JSLExceptionCode.UNSUPPORTED_OPERATION, message, cause);
    }
}
