package basalt;

/**
 * Unified unchecked exception for the Basalt SDK.
 *
 * <p>Native Rust/C++ panics, JNI shape rejections, memory errors, and invalid arguments are
 * translated into this hierarchy. It extends {@link IllegalStateException} so existing code that
 * catches the JDK exception keeps working, while SDK users can inspect {@link #errorCode()} or
 * catch specific subclasses ({@link BasaltParameterException}, {@link BasaltShapeException},
 * {@link BasaltMemoryException}, {@link BasaltUnsupportedOperationException}).</p>
 */
public class BasaltRuntimeException extends IllegalStateException {
    private static final long serialVersionUID = 1L;

    private final BasaltExceptionCode errorCode;

    public BasaltRuntimeException(String message) {
        this(BasaltExceptionCode.INVALID_PARAMETER, message);
    }

    public BasaltRuntimeException(BasaltExceptionCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode == null ? BasaltExceptionCode.INVALID_PARAMETER : errorCode;
    }

    public BasaltRuntimeException(String message, Throwable cause) {
        this(BasaltExceptionCode.INVALID_PARAMETER, message, cause);
    }

    public BasaltRuntimeException(BasaltExceptionCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode == null ? BasaltExceptionCode.INVALID_PARAMETER : errorCode;
    }

    public BasaltExceptionCode errorCode() {
        return errorCode;
    }

    public int statusCode() {
        return errorCode.code();
    }

    public static BasaltRuntimeException fromStatus(int status, String message) {
        BasaltExceptionCode code = BasaltExceptionCode.fromCode(status);
        String fullMessage = message == null || message.isBlank()
                ? code.description() + " (code " + status + ")"
                : message;
        return switch (code) {
            case INVALID_PARAMETER -> new BasaltParameterException(fullMessage);
            case SHAPE_MISMATCH -> new BasaltShapeException(fullMessage);
            case OUT_OF_MEMORY -> new BasaltMemoryException(fullMessage);
            case UNSUPPORTED_OPERATION -> new BasaltUnsupportedOperationException(fullMessage);
            default -> new BasaltRuntimeException(code, fullMessage);
        };
    }

    public static <T> T requireNonNull(T value, String name) {
        if (value == null) {
            throw new BasaltParameterException(name + " must not be null");
        }
        return value;
    }
}
