package jsl;

/**
 * Unified unchecked exception for the JSL SDK.
 *
 * <p>Native Rust panics, JNI shape rejections and null arguments are translated into this type.
 * It extends {@link IllegalStateException} so existing code that catches the JDK exception keeps
 * working, while SDK users can catch one JSL-specific type.</p>
 */
public class JSLRuntimeException extends IllegalStateException {
    private static final long serialVersionUID = 1L;

    public JSLRuntimeException(String message) {
        super(message);
    }

    public JSLRuntimeException(String message, Throwable cause) {
        super(message, cause);
    }

    public static <T> T requireNonNull(T value, String name) {
        if (value == null) {
            throw new JSLRuntimeException(name + " must not be null");
        }
        return value;
    }
}
