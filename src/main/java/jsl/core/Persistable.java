package jsl.core;

/** A model that can serialize itself into the versioned .jsl binary container. */
public interface Persistable {
    ModelKind kind();

    void write(ModelWriter writer);
}
