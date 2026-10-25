package jsl;

/**
 * Thrown when an invalid parameter, hyperparameter, or argument is supplied to a JSL operator or estimator.
 */
public final class JSLParameterException extends JSLRuntimeException {
    private static final long serialVersionUID = 1L;

    public JSLParameterException(String message) {
        super(JSLExceptionCode.INVALID_PARAMETER, message);
    }

    public JSLParameterException(String message, Throwable cause) {
        super(JSLExceptionCode.INVALID_PARAMETER, message, cause);
    }
}
