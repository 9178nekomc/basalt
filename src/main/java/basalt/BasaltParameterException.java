package basalt;

/**
 * Thrown when an invalid parameter, hyperparameter, or argument is supplied to a Basalt operator or estimator.
 */
public final class BasaltParameterException extends BasaltRuntimeException {
    private static final long serialVersionUID = 1L;

    public BasaltParameterException(String message) {
        super(BasaltExceptionCode.INVALID_PARAMETER, message);
    }

    public BasaltParameterException(String message, Throwable cause) {
        super(BasaltExceptionCode.INVALID_PARAMETER, message, cause);
    }
}
