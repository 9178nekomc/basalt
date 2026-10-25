package jsl.core;

/** Compile-time marker for a tensor's logical dimension order. */
public sealed interface Layout permits BCHW, BHW, BC, BSF {
}
