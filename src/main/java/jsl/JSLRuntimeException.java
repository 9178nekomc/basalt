package jsl;

/**
 * Unified unchecked exception for the JSL SDK.
 *
 * <p>Native Rust/C++ panics, JNI shape rejections, memory errors, and invalid arguments are
 * translated into this hierarchy. It extends {@link IllegalStateException} so existing code that
 * catches the JDK exception keeps working, while SDK users can inspect {@link #errorCode()} or
 * catch specific subclasses ({@link JSLParameterException}, {@link JSLShapeException},
 * {@link JSLMemoryException}, {@link JSLUnsupportedOperationException}).</p>
 */
public class JSLRuntimeException extends IllegalStateException {
    private static final long serialVersionUID = 1L;

    private final JSLExceptionCode errorCode;

    public JSLRuntimeException(String message) {
        this(JSLExceptionCode.INVALID_PARAMETER, message);
    }

    public JSLRuntimeException(JSLExceptionCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode == null ? JSLExceptionCode.INVALID_PARAMETER : errorCode;
    }

    public JSLRuntimeException(String message, Throwable cause) {
        this(JSLExceptionCode.INVALID_PARAMETER, message, cause);
    }

    public JSLRuntimeException(JSLExceptionCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode == null ? JSLExceptionCode.INVALID_PARAMETER : errorCode;
    }

    public JSLExceptionCode errorCode() {
        return errorCode;
    }

    public int statusCode() {
        return errorCode.code();
    }

    public static JSLRuntimeException fromStatus(int status, String message) {
        JSLExceptionCode code = JSLExceptionCode.fromCode(status);
        String fullMessage = message == null || message.isBlank()
                ? code.description() + " (code " + status + ")"
                : message;
        return switch (code) {
            case INVALID_PARAMETER -> new JSLParameterException(fullMessage);
            case SHAPE_MISMATCH -> new JSLShapeException(fullMessage);
            case OUT_OF_MEMORY -> new JSLMemoryException(fullMessage);
            case UNSUPPORTED_OPERATION -> new JSLUnsupportedOperationException(fullMessage);
            default -> new JSLRuntimeException(code, fullMessage);
        };
    }

    public static <T> T requireNonNull(T value, String name) {
        if (value == null) {
            throw new JSLParameterException(name + " must not be null");
        }
        return value;
    }
}
