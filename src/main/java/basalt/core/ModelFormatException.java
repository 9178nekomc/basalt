package basalt.core;

/** Unchecked failure raised when a .bsl stream is truncated, corrupt or incompatible. */
public final class ModelFormatException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public ModelFormatException(String message) {
        super(message);
    }

    public ModelFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
