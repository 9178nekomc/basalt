package basalt.core;

/** A model that can serialize itself into the versioned .bsl binary container. */
public interface Persistable {
    ModelKind kind();

    void write(ModelWriter writer);
}
