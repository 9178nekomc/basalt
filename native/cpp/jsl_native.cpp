#include "jni.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jsl {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID = -1;
constexpr int32_t STATUS_SHAPE = -2;
constexpr int32_t STATUS_UNKNOWN_TENSOR = -3;
constexpr int32_t STATUS_CONTEXT = -4;
constexpr int32_t STATUS_GRAPH = -5;
constexpr int32_t STATUS_PANIC = -6;
constexpr int32_t STATUS_OOM = -7;
constexpr int32_t STATUS_UNSUPPORTED = -8;
constexpr int32_t VERSION = 2;

struct NativeError : public std::exception {
    int32_t code;
    std::string message;
    NativeError(int32_t c, std::string m) : code(c), message(std::move(m)) {}
    const char* what() const noexcept override { return message.c_str(); }
    static NativeError invalid(const std::string& m) { return NativeError(STATUS_INVALID, m); }
    static NativeError shape(const std::string& m) { return NativeError(STATUS_SHAPE, m); }
    static NativeError graph(const std::string& m) { return NativeError(STATUS_GRAPH, m); }
    static NativeError oom(const std::string& m) { return NativeError(STATUS_OOM, m); }
    static NativeError unsupported(const std::string& m) { return NativeError(STATUS_UNSUPPORTED, m); }
    static NativeError unknown_tensor(uint64_t h) {
        std::ostringstream oss;
        oss << "tensor handle 0x" << std::hex << h << " is not live in this context";
        return NativeError(STATUS_UNKNOWN_TENSOR, oss.str());
    }
    const char* exception_class() const {
        switch (code) {
            case STATUS_INVALID: return "jsl/JSLParameterException";
            case STATUS_SHAPE: return "jsl/JSLShapeException";
            case STATUS_OOM: return "jsl/JSLMemoryException";
            case STATUS_UNSUPPORTED: return "jsl/JSLUnsupportedOperationException";
            default: return "jsl/JSLRuntimeException";
        }
    }
    std::string formatted() const {
        std::ostringstream oss;
        oss << message << " (code " << code << ")";
        return oss.str();
    }
};

static void throw_java_error(JNIEnv* env, const NativeError& err) {
    jclass cls = env->FindClass(err.exception_class());
    if (!cls) {
        cls = env->FindClass("jsl/JSLRuntimeException");
    }
    if (!cls) {
        cls = env->FindClass("java/lang/IllegalStateException");
    }
    if (cls) {
        std::string msg = err.formatted();
        env->ThrowNew(cls, msg.c_str());
    }
}

template <typename T, typename F>
static T run_jni(JNIEnv* env, T default_val, F&& action) {
    try {
        return action();
    } catch (const NativeError& err) {
        throw_java_error(env, err);
        return default_val;
    } catch (const std::bad_alloc&) {
        NativeError err = NativeError::oom("native memory allocation failed");
        throw_java_error(env, err);
        return default_val;
    } catch (const std::exception& ex) {
        NativeError err(STATUS_PANIC, std::string("native panic caught at the JNI boundary: ") + ex.what());
        throw_java_error(env, err);
        return default_val;
    } catch (...) {
        NativeError err(STATUS_PANIC, "native panic caught at the JNI boundary");
        throw_java_error(env, err);
        return default_val;
    }
}

static std::atomic<uint64_t> NEXT_TENSOR_HANDLE{0x10000001ULL};
static std::atomic<uint64_t> NEXT_CONTEXT_HANDLE{0x20000001ULL};
static std::atomic<uint64_t> NEXT_SESSION_HANDLE{0x30000001ULL};

static size_t checked_numel(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw NativeError::shape("tensor shape must not be empty");
    }
    size_t elements = 1;
    for (size_t d : shape) {
        if (d == 0) {
            throw NativeError::shape("tensor dimensions must be positive");
        }
        if (elements > (std::numeric_limits<size_t>::max() / d)) {
            throw NativeError::oom("tensor element count overflows usize");
        }
        elements *= d;
    }
    if (elements > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw NativeError::oom("tensor allocation exceeds the addressable byte range");
    }
    return elements;
}

struct NativeTensor {
    mutable std::shared_mutex mtx;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<size_t> shape;
    bool requires_grad;

    NativeTensor(std::vector<size_t> s, bool rg) : shape(std::move(s)), requires_grad(rg) {
        size_t n = checked_numel(shape);
        data.assign(n, 0.0f);
    }

    NativeTensor(std::vector<size_t> s, std::vector<float> d, bool rg)
        : data(std::move(d)), shape(std::move(s)), requires_grad(rg) {
        size_t n = checked_numel(shape);
        if (data.size() != n) {
            throw NativeError::shape("data length does not match shape");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            if (!std::isfinite(data[i])) {
                throw NativeError::invalid("tensor data contains a non-finite value");
            }
        }
    }

    size_t bytes() const {
        return (data.capacity() + grad.capacity()) * sizeof(float);
    }

    void ensure_grad() {
        if (grad.size() != data.size()) {
            grad.assign(data.size(), 0.0f);
        }
    }

    void zero_grad() {
        if (!grad.empty()) {
            std::fill(grad.begin(), grad.end(), 0.0f);
        }
    }

    void accumulate_grad(const std::vector<float>& g) {
        if (g.size() != data.size()) {
            throw NativeError::shape("gradient length mismatch");
        }
        ensure_grad();
        for (size_t i = 0; i < data.size(); ++i) {
            grad[i] += g[i];
        }
    }

    void set_data(const std::vector<float>& vals) {
        if (vals.size() != data.size()) {
            throw NativeError::shape("data length does not match tensor length");
        }
        for (float v : vals) {
            if (!std::isfinite(v)) {
                throw NativeError::invalid("tensor data contains a non-finite value");
            }
        }
        std::copy(vals.begin(), vals.end(), data.begin());
    }
};

enum class Op : uint8_t {
    MatMul = 1,
    Add = 2,
    ReLU = 3,
    Sub = 4,
    Mul = 5,
    Sigmoid = 6,
    Tanh = 7,
    Mse = 8,
    Softmax = 9,
    CrossEntropy = 10,
    Gelu = 11,
    Silu = 12,
    LayerNorm = 13,
    BatchNorm2d = 14,
    Conv2d = 15,
    MaxPool2d = 16,
    Dropout = 17,
    Embedding = 18,
    BatchedMatMul = 19,
    Permute = 20,
    Reshape = 21,
    GroupNorm = 22
};

static Op op_from_code(uint8_t code) {
    if (code >= 1 && code <= 22) {
        return static_cast<Op>(code);
    }
    std::ostringstream oss;
    oss << "unknown op code " << static_cast<int>(code) << "; expected 1..=22";
    throw NativeError::unsupported(oss.str());
}

static size_t op_arity(Op op) {
    switch (op) {
        case Op::ReLU:
        case Op::Sigmoid:
        case Op::Tanh:
        case Op::Gelu:
        case Op::Silu:
        case Op::Softmax:
        case Op::MaxPool2d:
        case Op::Dropout:
        case Op::Permute:
        case Op::Reshape:
            return 1;
        case Op::MatMul:
        case Op::Add:
        case Op::Sub:
        case Op::Mul:
        case Op::Mse:
        case Op::CrossEntropy:
        case Op::Conv2d:
        case Op::Embedding:
        case Op::BatchedMatMul:
            return 2;
        case Op::LayerNorm:
        case Op::GroupNorm:
            return 3;
        case Op::BatchNorm2d:
            return 5;
    }
    return 1;
}

struct Node {
    Op op;
    std::vector<uint64_t> inputs;
    uint64_t output;
    size_t m = 0, k = 0, n = 0, batch = 0;
    size_t rows = 0, dim = 0, classes = 0;
    size_t c = 0, h = 0, w = 0, cout = 0, kh = 0, kw = 0, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1, groups = 1, oh = 0, ow = 0, spatial = 0;
    float eps = 1e-5f, momentum = 0.1f, probability = 0.0f;
    bool training = false;
    uint64_t seed = 0;
    std::vector<size_t> a_shape, b_shape, in_shape, out_shape, perm;
};

struct Graph {
    std::vector<Node> nodes;
    std::unordered_map<uint64_t, size_t> produced;
    std::unordered_set<uint64_t> consumed;

    void clear() {
        nodes.clear();
        produced.clear();
        consumed.clear();
    }

    bool uses(uint64_t addr) const {
        return produced.count(addr) > 0 || consumed.count(addr) > 0;
    }

    void check_push(Op op, const std::vector<uint64_t>& inputs, uint64_t output) const {
        if (inputs.size() != op_arity(op)) {
            throw NativeError::graph("operator input arity mismatch");
        }
        for (uint64_t in : inputs) {
            if (in == output) {
                throw NativeError::graph("cannot write into one of its inputs (in-place ops are not differentiable)");
            }
        }
        if (produced.count(output) > 0) {
            throw NativeError::graph("tensor is already produced by the current graph; call clearGraph between steps");
        }
        if (consumed.count(output) > 0) {
            throw NativeError::graph("tensor was already consumed as an input; overwriting it would break the graph");
        }
    }

    void push_unchecked(Node node) {
        size_t idx = nodes.size();
        produced[node.output] = idx;
        for (uint64_t in : node.inputs) {
            consumed.insert(in);
        }
        nodes.push_back(std::move(node));
    }

    std::vector<size_t> topological_order(uint64_t root) const {
        auto it = produced.find(root);
        if (it == produced.end()) {
            throw NativeError::graph("backward root was not produced by the current graph");
        }
        size_t root_idx = it->second;
        std::vector<bool> reachable(nodes.size(), false);
        std::vector<size_t> stack{root_idx};
        while (!stack.empty()) {
            size_t idx = stack.back();
            stack.pop_back();
            if (reachable[idx]) continue;
            reachable[idx] = true;
            for (uint64_t in : nodes[idx].inputs) {
                auto pit = produced.find(in);
                if (pit != produced.end() && !reachable[pit->second]) {
                    stack.push_back(pit->second);
                }
            }
        }
        std::vector<size_t> indegree(nodes.size(), 0);
        std::unordered_map<size_t, std::vector<size_t>> consumers;
        size_t expected = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (!reachable[i]) continue;
            expected++;
            for (uint64_t in : nodes[i].inputs) {
                auto pit = produced.find(in);
                if (pit != produced.end() && reachable[pit->second]) {
                    indegree[i]++;
                    consumers[pit->second].push_back(i);
                }
            }
        }
        std::queue<size_t> q;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (reachable[i] && indegree[i] == 0) {
                q.push(i);
            }
        }
        std::vector<size_t> order;
        order.reserve(expected);
        while (!q.empty()) {
            size_t u = q.front();
            q.pop();
            order.push_back(u);
            auto cit = consumers.find(u);
            if (cit != consumers.end()) {
                for (size_t v : cit->second) {
                    if (--indegree[v] == 0) {
                        q.push(v);
                    }
                }
            }
        }
        if (order.size() != expected) {
            throw NativeError::graph("cycle detected while topologically sorting the autodiff graph");
        }
        return order;
    }
};

struct AdamState {
    std::vector<float> first;
    std::vector<float> second;
};

struct Context {
    mutable std::shared_mutex tensors_mtx;
    std::unordered_map<uint64_t, std::shared_ptr<NativeTensor>> tensors;
    size_t live_bytes = 0;

    mutable std::mutex graph_mtx;
    Graph graph;

    mutable std::mutex opt_mtx;
    std::unordered_map<uint64_t, AdamState> optimizers;

    uint64_t create_tensor(std::vector<size_t> shape, bool requires_grad) {
        auto t = std::make_shared<NativeTensor>(std::move(shape), requires_grad);
        uint64_t h = NEXT_TENSOR_HANDLE.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::shared_mutex> lk(tensors_mtx);
        live_bytes += t->bytes();
        tensors[h] = std::move(t);
        return h;
    }

    uint64_t create_tensor_from_data(std::vector<size_t> shape, std::vector<float> data, bool requires_grad) {
        auto t = std::make_shared<NativeTensor>(std::move(shape), std::move(data), requires_grad);
        uint64_t h = NEXT_TENSOR_HANDLE.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock<std::shared_mutex> lk(tensors_mtx);
        live_bytes += t->bytes();
        tensors[h] = std::move(t);
        return h;
    }

    std::shared_ptr<NativeTensor> tensor(uint64_t h) const {
        std::shared_lock<std::shared_mutex> lk(tensors_mtx);
        auto it = tensors.find(h);
        if (it == tensors.end()) {
            throw NativeError::unknown_tensor(h);
        }
        return it->second;
    }

    bool release_tensor(uint64_t h) {
        {
            std::lock_guard<std::mutex> glk(graph_mtx);
            if (graph.uses(h)) {
                graph.clear();
            }
        }
        bool removed = false;
        {
            std::unique_lock<std::shared_mutex> lk(tensors_mtx);
            auto it = tensors.find(h);
            if (it != tensors.end()) {
                size_t b = it->second->bytes();
                live_bytes = live_bytes >= b ? live_bytes - b : 0;
                tensors.erase(it);
                removed = true;
            }
        }
        if (removed) {
            std::lock_guard<std::mutex> olk(opt_mtx);
            optimizers.erase(h);
        }
        return removed;
    }

    void stats(int64_t out[3]) const {
        std::shared_lock<std::shared_mutex> tlk(tensors_mtx);
        std::lock_guard<std::mutex> glk(graph_mtx);
        out[0] = static_cast<int64_t>(tensors.size());
        out[1] = static_cast<int64_t>(live_bytes);
        out[2] = static_cast<int64_t>(graph.nodes.size());
    }
};

static std::mutex REGISTRY_MTX;
static std::unordered_map<uint64_t, std::shared_ptr<Context>> CONTEXT_REGISTRY;

static std::shared_ptr<Context> context_from_ptr(jlong ptr) {
    if (ptr == 0) {
        throw NativeError(STATUS_CONTEXT, "null context handle");
    }
    uint64_t h = static_cast<uint64_t>(ptr);
    std::lock_guard<std::mutex> lk(REGISTRY_MTX);
    auto it = CONTEXT_REGISTRY.find(h);
    if (it == CONTEXT_REGISTRY.end()) {
        std::ostringstream oss;
        oss << "context 0x" << std::hex << h << " is not live (already closed?)";
        throw NativeError(STATUS_CONTEXT, oss.str());
    }
    return it->second;
}

#pragma pack(push, 1)
struct OpCommand {
    uint8_t op_code;
    uint8_t flags;
    uint8_t _padding[6];
    uint64_t input_a;
    uint64_t input_b;
    uint64_t input_c;
    uint64_t input_d;
    uint64_t input_e;
    uint64_t output;
    int32_t attrs[16];
    uint8_t _reserved[8];
};
#pragma pack(pop)

static_assert(sizeof(OpCommand) == 128, "OpCommand must be 128 bytes");

static float attr_f32(const OpCommand& cmd, size_t idx) {
    uint32_t bits = static_cast<uint32_t>(cmd.attrs[idx]);
    float val;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

static uint64_t attr_u64(const OpCommand& cmd, size_t low, size_t high) {
    return static_cast<uint64_t>(static_cast<uint32_t>(cmd.attrs[low]))
         | (static_cast<uint64_t>(static_cast<uint32_t>(cmd.attrs[high])) << 32);
}

static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        size_t ad = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        size_t bd = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (ad == bd) out[i] = ad;
        else if (ad == 1) out[i] = bd;
        else if (bd == 1) out[i] = ad;
        else throw NativeError::shape("cannot broadcast shapes");
    }
    return out;
}

static std::vector<size_t> aligned_strides(const std::vector<size_t>& shape, const std::vector<size_t>& out_shape) {
    size_t offset = out_shape.size() - shape.size();
    std::vector<size_t> strides(out_shape.size(), 0);
    size_t s = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        bool bcast = (shape[i] == 1 && out_shape[offset + i] != 1);
        strides[offset + i] = bcast ? 0 : s;
        s *= shape[i];
    }
    return strides;
}

static void gemm_f32(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    std::fill(c, c + m * n, 0.0f);
    for (size_t i = 0; i < m; ++i) {
        float* c_row = c + i * n;
        const float* a_row = a + i * k;
        for (size_t p = 0; p < k; ++p) {
            float alpha = a_row[p];
            if (alpha != 0.0f) {
                const float* b_row = b + p * n;
                for (size_t j = 0; j < n; ++j) {
                    c_row[j] += alpha * b_row[j];
                }
            }
        }
    }
}

static void gemm_nt(const float* grad_c, const float* b, float* grad_a, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        const float* gc_row = grad_c + i * n;
        for (size_t p = 0; p < k; ++p) {
            const float* b_row = b + p * n;
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
                sum += gc_row[j] * b_row[j];
            }
            grad_a[i * k + p] += sum;
        }
    }
}

static void gemm_tn(const float* a, const float* grad_c, float* grad_b, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        const float* a_row = a + i * k;
        const float* gc_row = grad_c + i * n;
        for (size_t p = 0; p < k; ++p) {
            float alpha = a_row[p];
            if (alpha != 0.0f) {
                float* gb_row = grad_b + p * n;
                for (size_t j = 0; j < n; ++j) {
                    gb_row[j] += alpha * gc_row[j];
                }
            }
        }
    }
}

static float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        float e = std::exp(x);
        return e / (1.0f + e);
    }
}

static std::vector<float> dropout_mask(size_t length, float probability, bool training, uint64_t seed) {
    if (!training || probability <= 0.0f) {
        return std::vector<float>(length, 1.0f);
    }
    uint64_t state = seed | 1ULL;
    float scale = 1.0f / (1.0f - probability);
    std::vector<float> mask(length, 0.0f);
    for (size_t i = 0; i < length; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        float unit = static_cast<float>(state >> 40) / static_cast<float>(1ULL << 24);
        mask[i] = (unit >= probability) ? scale : 0.0f;
    }
    return mask;
}

static void execute_command(Context& ctx, Graph& graph, const OpCommand& cmd) {
    Op op = op_from_code(cmd.op_code);
    if (cmd.output == 0) {
        throw NativeError::invalid("op output handle must not be null");
    }
    uint64_t cand[5] = {cmd.input_a, cmd.input_b, cmd.input_c, cmd.input_d, cmd.input_e};
    size_t arity = op_arity(op);
    std::vector<uint64_t> inputs;
    inputs.reserve(arity);
    for (size_t i = 0; i < arity; ++i) {
        if (cand[i] == 0) {
            throw NativeError::invalid("command input handle is null");
        }
        inputs.push_back(cand[i]);
    }
    graph.check_push(op, inputs, cmd.output);
    auto out_t = ctx.tensor(cmd.output);
    std::vector<size_t> out_shape;
    size_t out_len = 0;
    {
        std::shared_lock<std::shared_mutex> lk(out_t->mtx);
        out_shape = out_t->shape;
        out_len = out_t->data.size();
    }

    switch (op) {
        case Op::MatMul: {
            auto left = ctx.tensor(cmd.input_a);
            auto right = ctx.tensor(cmd.input_b);
            if (left->shape.size() != 2 || right->shape.size() != 2) {
                throw NativeError::shape("MatMul expects rank-2 operands");
            }
            size_t m = left->shape[0], k = left->shape[1];
            size_t k2 = right->shape[0], n = right->shape[1];
            if (k != k2 || out_shape != std::vector<size_t>{m, n}) {
                throw NativeError::shape("MatMul shape mismatch");
            }
            gemm_f32(left->data.data(), right->data.data(), out_t->data.data(), m, k, n);
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.m = m; node.k = k; node.n = n;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Add:
        case Op::Sub:
        case Op::Mul: {
            auto left = ctx.tensor(cmd.input_a);
            auto right = ctx.tensor(cmd.input_b);
            auto bcast = broadcast_shape(left->shape, right->shape);
            if (out_shape != bcast) {
                throw NativeError::shape("binary broadcast output shape mismatch");
            }
            size_t rank = out_shape.size();
            auto a_str = aligned_strides(left->shape, out_shape);
            auto b_str = aligned_strides(right->shape, out_shape);
            std::vector<size_t> coords(rank, 0);
            for (size_t idx = 0; idx < out_len; ++idx) {
                size_t ai = 0, bi = 0;
                for (size_t d = 0; d < rank; ++d) {
                    ai += coords[d] * a_str[d];
                    bi += coords[d] * b_str[d];
                }
                if (op == Op::Add) out_t->data[idx] = left->data[ai] + right->data[bi];
                else if (op == Op::Sub) out_t->data[idx] = left->data[ai] - right->data[bi];
                else out_t->data[idx] = left->data[ai] * right->data[bi];
                for (size_t d = rank; d-- > 0;) {
                    if (++coords[d] < out_shape[d]) break;
                    coords[d] = 0;
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.a_shape = left->shape; node.b_shape = right->shape;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::ReLU:
        case Op::Sigmoid:
        case Op::Tanh:
        case Op::Gelu:
        case Op::Silu: {
            auto in_t = ctx.tensor(cmd.input_a);
            if (out_shape != in_t->shape) {
                throw NativeError::shape("unary activation output shape mismatch");
            }
            const float c_gelu = std::sqrt(2.0f / 3.14159265358979323846f);
            for (size_t i = 0; i < out_len; ++i) {
                float v = in_t->data[i];
                if (op == Op::ReLU) out_t->data[i] = std::max(0.0f, v);
                else if (op == Op::Sigmoid) out_t->data[i] = stable_sigmoid(v);
                else if (op == Op::Tanh) out_t->data[i] = std::tanh(v);
                else if (op == Op::Gelu) {
                    float inner = c_gelu * (v + 0.044715f * v * v * v);
                    out_t->data[i] = 0.5f * v * (1.0f + std::tanh(inner));
                } else {
                    out_t->data[i] = v * stable_sigmoid(v);
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Mse: {
            auto pred = ctx.tensor(cmd.input_a);
            auto targ = ctx.tensor(cmd.input_b);
            if (pred->shape != targ->shape || out_shape != std::vector<size_t>{1}) {
                throw NativeError::shape("MSE expects equal shapes and scalar output");
            }
            double sum = 0.0;
            for (size_t i = 0; i < pred->data.size(); ++i) {
                double d = static_cast<double>(pred->data[i]) - static_cast<double>(targ->data[i]);
                sum += d * d;
            }
            out_t->data[0] = static_cast<float>(sum / pred->data.size());
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Softmax: {
            auto in_t = ctx.tensor(cmd.input_a);
            size_t dim = cmd.attrs[0] > 0 ? static_cast<size_t>(cmd.attrs[0]) : 0;
            if (dim == 0 || in_t->shape.empty() || in_t->shape.back() != dim || out_shape != in_t->shape) {
                throw NativeError::shape("softmax expects last dimension to match");
            }
            size_t rows = out_len / dim;
            for (size_t r = 0; r < rows; ++r) {
                size_t base = r * dim;
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < dim; ++j) mx = std::max(mx, in_t->data[base + j]);
                float sum = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    float e = std::exp(in_t->data[base + j] - mx);
                    out_t->data[base + j] = e;
                    sum += e;
                }
                float inv = 1.0f / sum;
                for (size_t j = 0; j < dim; ++j) out_t->data[base + j] *= inv;
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.rows = rows; node.dim = dim;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::CrossEntropy: {
            auto logits = ctx.tensor(cmd.input_a);
            auto targets = ctx.tensor(cmd.input_b);
            if (logits->shape.size() != 2) {
                throw NativeError::shape("cross-entropy expects rank-2 logits");
            }
            size_t rows = logits->shape[0], classes = logits->shape[1];
            if (targets->data.size() != rows || out_shape != std::vector<size_t>{1}) {
                throw NativeError::shape("cross-entropy target/output shape mismatch");
            }
            double total = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                float t = targets->data[r];
                if (!std::isfinite(t) || t < 0.0f || t >= static_cast<float>(classes)) {
                    throw NativeError::invalid("cross-entropy target out of range");
                }
                size_t base = r * classes;
                float mx = -std::numeric_limits<float>::infinity();
                for (size_t c = 0; c < classes; ++c) mx = std::max(mx, logits->data[base + c]);
                double sum = 0.0;
                for (size_t c = 0; c < classes; ++c) sum += std::exp(static_cast<double>(logits->data[base + c] - mx));
                total += (static_cast<double>(mx) + std::log(sum)) - static_cast<double>(logits->data[base + static_cast<size_t>(t)]);
            }
            out_t->data[0] = static_cast<float>(total / rows);
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.rows = rows; node.classes = classes;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::LayerNorm: {
            auto in_t = ctx.tensor(cmd.input_a);
            auto gamma = ctx.tensor(cmd.input_b);
            auto beta = ctx.tensor(cmd.input_c);
            size_t dim = in_t->shape.empty() ? 0 : in_t->shape.back();
            if (dim == 0 || out_shape != in_t->shape || gamma->data.size() != dim || beta->data.size() != dim) {
                throw NativeError::shape("layer-norm shape mismatch");
            }
            size_t rows = out_len / dim;
            float eps = attr_f32(cmd, 0);
            for (size_t r = 0; r < rows; ++r) {
                size_t base = r * dim;
                double mean = 0.0;
                for (size_t j = 0; j < dim; ++j) mean += in_t->data[base + j];
                mean /= dim;
                double var = 0.0;
                for (size_t j = 0; j < dim; ++j) {
                    double d = in_t->data[base + j] - mean;
                    var += d * d;
                }
                var /= dim;
                double inv = 1.0 / std::sqrt(var + eps);
                for (size_t j = 0; j < dim; ++j) {
                    float norm = static_cast<float>((in_t->data[base + j] - mean) * inv);
                    out_t->data[base + j] = norm * gamma->data[j] + beta->data[j];
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.rows = rows; node.dim = dim; node.eps = eps;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::GroupNorm: {
            auto in_t = ctx.tensor(cmd.input_a);
            auto gamma = ctx.tensor(cmd.input_b);
            auto beta = ctx.tensor(cmd.input_c);
            if (in_t->shape.size() < 2) {
                throw NativeError::shape("group-norm expects rank >= 2 input");
            }
            size_t n = in_t->shape[0];
            size_t c = in_t->shape[1];
            size_t spatial = 1;
            for (size_t i = 2; i < in_t->shape.size(); ++i) spatial *= in_t->shape[i];
            size_t groups = cmd.attrs[0] > 0 ? static_cast<size_t>(cmd.attrs[0]) : 1;
            float eps = attr_f32(cmd, 1);
            if (c % groups != 0 || out_shape != in_t->shape || gamma->data.size() != c || beta->data.size() != c) {
                throw NativeError::shape("group-norm shape mismatch");
            }
            size_t c_per_g = c / groups;
            size_t group_elems = c_per_g * spatial;
            for (size_t b = 0; b < n; ++b) {
                for (size_t g = 0; g < groups; ++g) {
                    size_t c_start = g * c_per_g;
                    double mean = 0.0;
                    for (size_t ch = 0; ch < c_per_g; ++ch) {
                        size_t base = ((b * c) + c_start + ch) * spatial;
                        for (size_t s = 0; s < spatial; ++s) mean += in_t->data[base + s];
                    }
                    mean /= group_elems;
                    double var = 0.0;
                    for (size_t ch = 0; ch < c_per_g; ++ch) {
                        size_t base = ((b * c) + c_start + ch) * spatial;
                        for (size_t s = 0; s < spatial; ++s) {
                            double d = in_t->data[base + s] - mean;
                            var += d * d;
                        }
                    }
                    var /= group_elems;
                    double inv = 1.0 / std::sqrt(var + eps);
                    for (size_t ch = 0; ch < c_per_g; ++ch) {
                        size_t channel = c_start + ch;
                        size_t base = ((b * c) + channel) * spatial;
                        for (size_t s = 0; s < spatial; ++s) {
                            float x_hat = static_cast<float>((in_t->data[base + s] - mean) * inv);
                            out_t->data[base + s] = x_hat * gamma->data[channel] + beta->data[channel];
                        }
                    }
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.n = n; node.c = c; node.spatial = spatial; node.groups = groups; node.eps = eps;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::BatchNorm2d: {
            auto in_t = ctx.tensor(cmd.input_a);
            auto gamma = ctx.tensor(cmd.input_b);
            auto beta = ctx.tensor(cmd.input_c);
            auto rmean = ctx.tensor(cmd.input_d);
            auto rvar = ctx.tensor(cmd.input_e);
            if (in_t->shape.size() != 4) throw NativeError::shape("batch-norm expects NCHW input");
            size_t n = in_t->shape[0], c = in_t->shape[1], h = in_t->shape[2], w = in_t->shape[3];
            if (out_shape != in_t->shape || gamma->data.size() != c || beta->data.size() != c
                || rmean->data.size() != c || rvar->data.size() != c) {
                throw NativeError::shape("batch-norm parameter shape mismatch");
            }
            float eps = attr_f32(cmd, 0);
            float momentum = attr_f32(cmd, 1);
            bool training = cmd.attrs[2] != 0;
            size_t spatial = n * h * w;
            for (size_t ch = 0; ch < c; ++ch) {
                if (training) {
                    double mean = 0.0;
                    for (size_t b = 0; b < n; ++b)
                        for (size_t r = 0; r < h; ++r)
                            for (size_t col = 0; col < w; ++col)
                                mean += in_t->data[((b * c + ch) * h + r) * w + col];
                    mean /= spatial;
                    double var = 0.0;
                    for (size_t b = 0; b < n; ++b)
                        for (size_t r = 0; r < h; ++r)
                            for (size_t col = 0; col < w; ++col) {
                                double d = in_t->data[((b * c + ch) * h + r) * w + col] - mean;
                                var += d * d;
                            }
                    var /= spatial;
                    rmean->data[ch] = static_cast<float>((1.0 - momentum) * rmean->data[ch] + momentum * mean);
                    double unbiased = spatial > 1 ? var * spatial / (spatial - 1) : var;
                    rvar->data[ch] = static_cast<float>((1.0 - momentum) * rvar->data[ch] + momentum * unbiased);
                    double inv = 1.0 / std::sqrt(var + eps);
                    for (size_t b = 0; b < n; ++b)
                        for (size_t r = 0; r < h; ++r)
                            for (size_t col = 0; col < w; ++col) {
                                size_t idx = ((b * c + ch) * h + r) * w + col;
                                float norm = static_cast<float>((in_t->data[idx] - mean) * inv);
                                out_t->data[idx] = norm * gamma->data[ch] + beta->data[ch];
                            }
                } else {
                    double inv = 1.0 / std::sqrt(rvar->data[ch] + eps);
                    for (size_t b = 0; b < n; ++b)
                        for (size_t r = 0; r < h; ++r)
                            for (size_t col = 0; col < w; ++col) {
                                size_t idx = ((b * c + ch) * h + r) * w + col;
                                float norm = static_cast<float>((in_t->data[idx] - rmean->data[ch]) * inv);
                                out_t->data[idx] = norm * gamma->data[ch] + beta->data[ch];
                            }
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.n = n; node.c = c; node.h = h; node.w = w; node.eps = eps; node.momentum = momentum; node.training = training;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Conv2d: {
            auto in_t = ctx.tensor(cmd.input_a);
            auto weight = ctx.tensor(cmd.input_b);
            if (in_t->shape.size() != 4 || weight->shape.size() != 4) {
                throw NativeError::shape("conv2d expects NCHW input and OIHW weight");
            }
            size_t n = in_t->shape[0], cin = in_t->shape[1], h = in_t->shape[2], w = in_t->shape[3];
            size_t cout = weight->shape[0], w_cin = weight->shape[1], kh = weight->shape[2], kw = weight->shape[3];
            size_t sh = std::max(1, cmd.attrs[2]);
            size_t sw = std::max(1, cmd.attrs[3]);
            size_t ph = std::max(0, cmd.attrs[4]);
            size_t pw = std::max(0, cmd.attrs[5]);
            size_t dh = std::max(1, cmd.attrs[6]);
            size_t dw = std::max(1, cmd.attrs[7]);
            size_t groups = std::max(1, cmd.attrs[8]);
            if (cin % groups != 0 || cout % groups != 0 || cin / groups != w_cin || kh == 0 || kw == 0) {
                throw NativeError::shape("conv2d channel/group/kernel mismatch");
            }
            size_t eff_kh = dh * (kh - 1) + 1;
            size_t eff_kw = dw * (kw - 1) + 1;
            if (h + 2 * ph < eff_kh || w + 2 * pw < eff_kw) {
                throw NativeError::shape("conv2d kernel exceeds padded input");
            }
            size_t oh = (h + 2 * ph - eff_kh) / sh + 1;
            size_t ow = (w + 2 * pw - eff_kw) / sw + 1;
            if (out_shape != std::vector<size_t>{n, cout, oh, ow}) {
                throw NativeError::shape("conv2d output shape mismatch");
            }
            size_t cin_per_g = cin / groups;
            size_t cout_per_g = cout / groups;
            size_t kernel_len = cin_per_g * kh * kw;
            for (size_t b = 0; b < n; ++b) {
                for (size_t g = 0; g < groups; ++g) {
                    for (size_t oc = 0; oc < cout_per_g; ++oc) {
                        size_t cout_idx = g * cout_per_g + oc;
                        size_t w_base = cout_idx * kernel_len;
                        for (size_t orow = 0; orow < oh; ++orow) {
                            for (size_t ocol = 0; ocol < ow; ++ocol) {
                                float sum = 0.0f;
                                for (size_t ic = 0; ic < cin_per_g; ++ic) {
                                    size_t cin_idx = g * cin_per_g + ic;
                                    size_t in_base = ((b * cin + cin_idx) * h) * w;
                                    size_t w_ic_base = w_base + ic * kh * kw;
                                    for (size_t kr = 0; kr < kh; ++kr) {
                                        int64_t ir = static_cast<int64_t>(orow * sh + kr * dh) - static_cast<int64_t>(ph);
                                        if (ir < 0 || ir >= static_cast<int64_t>(h)) continue;
                                        for (size_t kc = 0; kc < kw; ++kc) {
                                            int64_t ic_col = static_cast<int64_t>(ocol * sw + kc * dw) - static_cast<int64_t>(pw);
                                            if (ic_col < 0 || ic_col >= static_cast<int64_t>(w)) continue;
                                            sum += in_t->data[in_base + static_cast<size_t>(ir) * w + static_cast<size_t>(ic_col)]
                                                 * weight->data[w_ic_base + kr * kw + kc];
                                        }
                                    }
                                }
                                out_t->data[((b * cout + cout_idx) * oh + orow) * ow + ocol] = sum;
                            }
                        }
                    }
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.n = n; node.c = cin; node.h = h; node.w = w; node.cout = cout;
            node.kh = kh; node.kw = kw; node.sh = sh; node.sw = sw; node.ph = ph; node.pw = pw;
            node.dh = dh; node.dw = dw; node.groups = groups; node.oh = oh; node.ow = ow;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::MaxPool2d: {
            auto in_t = ctx.tensor(cmd.input_a);
            if (in_t->shape.size() != 4) throw NativeError::shape("max-pool expects NCHW input");
            size_t n = in_t->shape[0], c = in_t->shape[1], h = in_t->shape[2], w = in_t->shape[3];
            size_t kh = std::max(1, cmd.attrs[0]);
            size_t kw = std::max(1, cmd.attrs[1]);
            size_t sh = std::max(1, cmd.attrs[2]);
            size_t sw = std::max(1, cmd.attrs[3]);
            if (kh > h || kw > w) throw NativeError::shape("max-pool kernel larger than input");
            size_t oh = (h - kh) / sh + 1;
            size_t ow = (w - kw) / sw + 1;
            if (out_shape != std::vector<size_t>{n, c, oh, ow}) throw NativeError::shape("max-pool output mismatch");
            for (size_t b = 0; b < n; ++b)
                for (size_t ch = 0; ch < c; ++ch)
                    for (size_t orow = 0; orow < oh; ++orow)
                        for (size_t ocol = 0; ocol < ow; ++ocol) {
                            float mx = -std::numeric_limits<float>::infinity();
                            for (size_t kr = 0; kr < kh; ++kr) {
                                size_t ir = orow * sh + kr;
                                for (size_t kc = 0; kc < kw; ++kc) {
                                    size_t ic = ocol * sw + kc;
                                    mx = std::max(mx, in_t->data[((b * c + ch) * h + ir) * w + ic]);
                                }
                            }
                            out_t->data[((b * c + ch) * oh + orow) * ow + ocol] = mx;
                        }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.n = n; node.c = c; node.h = h; node.w = w; node.kh = kh; node.kw = kw;
            node.sh = sh; node.sw = sw; node.oh = oh; node.ow = ow;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Dropout: {
            auto in_t = ctx.tensor(cmd.input_a);
            if (out_shape != in_t->shape) throw NativeError::shape("dropout shape mismatch");
            float prob = attr_f32(cmd, 0);
            if (prob < 0.0f || prob >= 1.0f) throw NativeError::invalid("dropout probability must be in [0, 1)");
            bool training = cmd.attrs[1] != 0;
            uint64_t seed = attr_u64(cmd, 2, 3);
            auto mask = dropout_mask(out_len, prob, training, seed);
            for (size_t i = 0; i < out_len; ++i) out_t->data[i] = in_t->data[i] * mask[i];
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.probability = prob; node.training = training; node.seed = seed;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Embedding: {
            auto indices = ctx.tensor(cmd.input_a);
            auto weight = ctx.tensor(cmd.input_b);
            if (weight->shape.size() != 2) throw NativeError::shape("embedding weight must be rank 2");
            size_t rows = indices->data.size();
            size_t vocab = weight->shape[0], dim = weight->shape[1];
            auto expected = indices->shape;
            expected.push_back(dim);
            if (out_shape != expected) throw NativeError::shape("embedding output shape mismatch");
            for (size_t r = 0; r < rows; ++r) {
                float idx_f = indices->data[r];
                if (!std::isfinite(idx_f) || idx_f < 0.0f || idx_f >= static_cast<float>(vocab)) {
                    throw NativeError::invalid("embedding index out of range");
                }
                size_t idx = static_cast<size_t>(idx_f);
                std::copy(weight->data.begin() + idx * dim, weight->data.begin() + (idx + 1) * dim,
                          out_t->data.begin() + r * dim);
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.rows = rows; node.dim = dim;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::BatchedMatMul: {
            auto left = ctx.tensor(cmd.input_a);
            auto right = ctx.tensor(cmd.input_b);
            if (left->shape.size() != 3 || right->shape.size() != 3) {
                throw NativeError::shape("batched-matmul expects rank-3 operands");
            }
            size_t batch = left->shape[0], m = left->shape[1], k = left->shape[2];
            if (right->shape[0] != batch || right->shape[1] != k || out_shape != std::vector<size_t>{batch, m, right->shape[2]}) {
                throw NativeError::shape("batched-matmul shape mismatch");
            }
            size_t n = right->shape[2];
            for (size_t b = 0; b < batch; ++b) {
                gemm_f32(left->data.data() + b * m * k, right->data.data() + b * k * n,
                         out_t->data.data() + b * m * n, m, k, n);
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.batch = batch; node.m = m; node.k = k; node.n = n;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Permute: {
            auto in_t = ctx.tensor(cmd.input_a);
            size_t rank = cmd.attrs[0] > 0 ? static_cast<size_t>(cmd.attrs[0]) : 0;
            if (rank == 0 || rank != in_t->shape.size()) throw NativeError::shape("permute rank mismatch");
            std::vector<size_t> perm(rank);
            std::vector<size_t> expected(rank);
            std::vector<bool> seen(rank, false);
            for (size_t i = 0; i < rank; ++i) {
                size_t p = static_cast<size_t>(cmd.attrs[1 + i]);
                if (p >= rank || seen[p]) throw NativeError::shape("invalid permutation");
                seen[p] = true;
                perm[i] = p;
                expected[i] = in_t->shape[p];
            }
            if (out_shape != expected) throw NativeError::shape("permute output shape mismatch");
            std::vector<size_t> in_strides(rank, 1);
            for (size_t i = rank - 1; i-- > 0;) in_strides[i] = in_strides[i + 1] * in_t->shape[i + 1];
            std::vector<size_t> coords(rank, 0);
            for (size_t oi = 0; oi < out_len; ++oi) {
                size_t ii = 0;
                for (size_t d = 0; d < rank; ++d) ii += coords[d] * in_strides[perm[d]];
                out_t->data[oi] = in_t->data[ii];
                for (size_t d = rank; d-- > 0;) {
                    if (++coords[d] < out_shape[d]) break;
                    coords[d] = 0;
                }
            }
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.in_shape = in_t->shape; node.perm = perm;
            graph.push_unchecked(std::move(node));
            break;
        }
        case Op::Reshape: {
            auto in_t = ctx.tensor(cmd.input_a);
            size_t rank = cmd.attrs[0] > 0 ? static_cast<size_t>(cmd.attrs[0]) : 0;
            if (rank == 0 || rank > 4) throw NativeError::shape("reshape rank must be in [1, 4]");
            std::vector<size_t> rshape(rank);
            for (size_t i = 0; i < rank; ++i) {
                if (cmd.attrs[1 + i] <= 0) throw NativeError::shape("reshape dim must be positive");
                rshape[i] = static_cast<size_t>(cmd.attrs[1 + i]);
            }
            if (out_shape != rshape || out_len != in_t->data.size()) throw NativeError::shape("reshape mismatch");
            std::copy(in_t->data.begin(), in_t->data.end(), out_t->data.begin());
            Node node;
            node.op = op; node.inputs = inputs; node.output = cmd.output;
            node.in_shape = in_t->shape; node.out_shape = rshape;
            graph.push_unchecked(std::move(node));
            break;
        }
    }
}

static void accumulate_grad_map(std::unordered_map<uint64_t, std::vector<float>>& grads,
                                uint64_t handle, const std::vector<float>& g) {
    auto it = grads.find(handle);
    if (it == grads.end()) {
        grads[handle] = g;
    } else {
        for (size_t i = 0; i < g.size(); ++i) it->second[i] += g[i];
    }
}

static void backward_impl(Context& ctx, uint64_t root) {
    std::lock_guard<std::mutex> glk(ctx.graph_mtx);
    auto order = ctx.graph.topological_order(root);
    auto root_t = ctx.tensor(root);
    std::unordered_map<uint64_t, std::vector<float>> gradients;
    gradients[root] = std::vector<float>(root_t->data.size(), 1.0f);

    for (size_t oi = order.size(); oi-- > 0;) {
        const Node& node = ctx.graph.nodes[order[oi]];
        auto git = gradients.find(node.output);
        if (git == gradients.end()) continue;
        const std::vector<float>& grad_output = git->second;

        switch (node.op) {
            case Op::MatMul: {
                auto left = ctx.tensor(node.inputs[0]);
                auto right = ctx.tensor(node.inputs[1]);
                std::vector<float> gl(node.m * node.k, 0.0f);
                std::vector<float> gr(node.k * node.n, 0.0f);
                gemm_nt(grad_output.data(), right->data.data(), gl.data(), node.m, node.k, node.n);
                gemm_tn(left->data.data(), grad_output.data(), gr.data(), node.m, node.k, node.n);
                accumulate_grad_map(gradients, node.inputs[0], gl);
                accumulate_grad_map(gradients, node.inputs[1], gr);
                break;
            }
            case Op::Add:
            case Op::Sub:
            case Op::Mul: {
                auto left = ctx.tensor(node.inputs[0]);
                auto right = ctx.tensor(node.inputs[1]);
                std::vector<float> gl(left->data.size(), 0.0f);
                std::vector<float> gr(right->data.size(), 0.0f);
                auto out_shape = broadcast_shape(node.a_shape, node.b_shape);
                size_t rank = out_shape.size();
                auto a_str = aligned_strides(node.a_shape, out_shape);
                auto b_str = aligned_strides(node.b_shape, out_shape);
                std::vector<size_t> coords(rank, 0);
                for (size_t idx = 0; idx < grad_output.size(); ++idx) {
                    size_t ai = 0, bi = 0;
                    for (size_t d = 0; d < rank; ++d) {
                        ai += coords[d] * a_str[d];
                        bi += coords[d] * b_str[d];
                    }
                    float go = grad_output[idx];
                    if (node.op == Op::Add) {
                        gl[ai] += go; gr[bi] += go;
                    } else if (node.op == Op::Sub) {
                        gl[ai] += go; gr[bi] -= go;
                    } else {
                        gl[ai] += go * right->data[bi];
                        gr[bi] += go * left->data[ai];
                    }
                    for (size_t d = rank; d-- > 0;) {
                        if (++coords[d] < out_shape[d]) break;
                        coords[d] = 0;
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gl);
                accumulate_grad_map(gradients, node.inputs[1], gr);
                break;
            }
            case Op::ReLU:
            case Op::Sigmoid:
            case Op::Tanh:
            case Op::Gelu:
            case Op::Silu: {
                auto in_t = ctx.tensor(node.inputs[0]);
                auto out_t = ctx.tensor(node.output);
                std::vector<float> gi(in_t->data.size(), 0.0f);
                const float c_gelu = std::sqrt(2.0f / 3.14159265358979323846f);
                for (size_t i = 0; i < gi.size(); ++i) {
                    float v = in_t->data[i];
                    float o = out_t->data[i];
                    float go = grad_output[i];
                    if (node.op == Op::ReLU) gi[i] = v > 0.0f ? go : 0.0f;
                    else if (node.op == Op::Sigmoid) gi[i] = go * o * (1.0f - o);
                    else if (node.op == Op::Tanh) gi[i] = go * (1.0f - o * o);
                    else if (node.op == Op::Gelu) {
                        float inner = c_gelu * (v + 0.044715f * v * v * v);
                        float th = std::tanh(inner);
                        float first = 0.5f * (1.0f + th);
                        float deriv = 0.5f * (1.0f - th * th) * c_gelu * (1.0f + 3.0f * 0.044715f * v * v);
                        gi[i] = go * (first + v * deriv);
                    } else {
                        float sig = stable_sigmoid(v);
                        gi[i] = go * sig * (1.0f + v * (1.0f - sig));
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                break;
            }
            case Op::Mse: {
                auto pred = ctx.tensor(node.inputs[0]);
                auto targ = ctx.tensor(node.inputs[1]);
                std::vector<float> gp(pred->data.size(), 0.0f);
                std::vector<float> gt(targ->data.size(), 0.0f);
                float scale = 2.0f * grad_output[0] / static_cast<float>(pred->data.size());
                for (size_t i = 0; i < pred->data.size(); ++i) {
                    float g = (pred->data[i] - targ->data[i]) * scale;
                    gp[i] = g;
                    gt[i] = -g;
                }
                accumulate_grad_map(gradients, node.inputs[0], gp);
                accumulate_grad_map(gradients, node.inputs[1], gt);
                break;
            }
            case Op::Softmax: {
                auto out_t = ctx.tensor(node.output);
                std::vector<float> gi(out_t->data.size(), 0.0f);
                for (size_t r = 0; r < node.rows; ++r) {
                    size_t base = r * node.dim;
                    float dot = 0.0f;
                    for (size_t j = 0; j < node.dim; ++j) dot += grad_output[base + j] * out_t->data[base + j];
                    for (size_t j = 0; j < node.dim; ++j) {
                        gi[base + j] = out_t->data[base + j] * (grad_output[base + j] - dot);
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                break;
            }
            case Op::CrossEntropy: {
                auto logits = ctx.tensor(node.inputs[0]);
                auto targets = ctx.tensor(node.inputs[1]);
                std::vector<float> gl(node.rows * node.classes, 0.0f);
                float scale = grad_output[0] / static_cast<float>(node.rows);
                for (size_t r = 0; r < node.rows; ++r) {
                    size_t base = r * node.classes;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (size_t c = 0; c < node.classes; ++c) mx = std::max(mx, logits->data[base + c]);
                    float sum = 0.0f;
                    for (size_t c = 0; c < node.classes; ++c) {
                        float e = std::exp(logits->data[base + c] - mx);
                        gl[base + c] = e;
                        sum += e;
                    }
                    float inv = 1.0f / sum;
                    for (size_t c = 0; c < node.classes; ++c) gl[base + c] *= inv;
                    gl[base + static_cast<size_t>(targets->data[r])] -= 1.0f;
                    for (size_t c = 0; c < node.classes; ++c) gl[base + c] *= scale;
                }
                accumulate_grad_map(gradients, node.inputs[0], gl);
                break;
            }
            case Op::LayerNorm: {
                auto in_t = ctx.tensor(node.inputs[0]);
                auto gamma = ctx.tensor(node.inputs[1]);
                std::vector<float> gi(node.rows * node.dim, 0.0f);
                std::vector<float> gg(node.dim, 0.0f);
                std::vector<float> gb(node.dim, 0.0f);
                for (size_t r = 0; r < node.rows; ++r) {
                    size_t base = r * node.dim;
                    double mean = 0.0;
                    for (size_t j = 0; j < node.dim; ++j) mean += in_t->data[base + j];
                    mean /= node.dim;
                    double var = 0.0;
                    for (size_t j = 0; j < node.dim; ++j) {
                        double d = in_t->data[base + j] - mean;
                        var += d * d;
                    }
                    var /= node.dim;
                    double inv = 1.0 / std::sqrt(var + node.eps);
                    float sum_dy = 0.0f, sum_dy_yhat = 0.0f;
                    for (size_t j = 0; j < node.dim; ++j) {
                        float norm = static_cast<float>((in_t->data[base + j] - mean) * inv);
                        float dy = grad_output[base + j];
                        sum_dy += dy;
                        sum_dy_yhat += dy * norm;
                        gg[j] += dy * norm;
                        gb[j] += dy;
                    }
                    for (size_t j = 0; j < node.dim; ++j) {
                        float norm = static_cast<float>((in_t->data[base + j] - mean) * inv);
                        float dy = grad_output[base + j];
                        gi[base + j] = static_cast<float>(inv) / static_cast<float>(node.dim)
                            * (static_cast<float>(node.dim) * dy * gamma->data[j]
                               - sum_dy * gamma->data[j]
                               - norm * sum_dy_yhat * gamma->data[j]);
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                accumulate_grad_map(gradients, node.inputs[1], gg);
                accumulate_grad_map(gradients, node.inputs[2], gb);
                break;
            }
            case Op::GroupNorm: {
                auto in_t = ctx.tensor(node.inputs[0]);
                auto gamma = ctx.tensor(node.inputs[1]);
                std::vector<float> gi(node.n * node.c * node.spatial, 0.0f);
                std::vector<float> gg(node.c, 0.0f);
                std::vector<float> gb(node.c, 0.0f);
                size_t c_per_g = node.c / node.groups;
                size_t group_elems = c_per_g * node.spatial;
                for (size_t b = 0; b < node.n; ++b) {
                    for (size_t g = 0; g < node.groups; ++g) {
                        size_t c_start = g * c_per_g;
                        double mean = 0.0;
                        for (size_t ch = 0; ch < c_per_g; ++ch) {
                            size_t base = ((b * node.c) + c_start + ch) * node.spatial;
                            for (size_t s = 0; s < node.spatial; ++s) mean += in_t->data[base + s];
                        }
                        mean /= group_elems;
                        double var = 0.0;
                        for (size_t ch = 0; ch < c_per_g; ++ch) {
                            size_t base = ((b * node.c) + c_start + ch) * node.spatial;
                            for (size_t s = 0; s < node.spatial; ++s) {
                                double d = in_t->data[base + s] - mean;
                                var += d * d;
                            }
                        }
                        var /= group_elems;
                        float inv_std = static_cast<float>(1.0 / std::sqrt(var + node.eps));
                        float sum_dy_gamma = 0.0f, sum_dy_gamma_xhat = 0.0f;
                        for (size_t ch = 0; ch < c_per_g; ++ch) {
                            size_t channel = c_start + ch;
                            size_t base = ((b * node.c) + channel) * node.spatial;
                            float gm = gamma->data[channel];
                            for (size_t s = 0; s < node.spatial; ++s) {
                                float x_hat = static_cast<float>(in_t->data[base + s] - mean) * inv_std;
                                float dy = grad_output[base + s];
                                gg[channel] += dy * x_hat;
                                gb[channel] += dy;
                                sum_dy_gamma += dy * gm;
                                sum_dy_gamma_xhat += dy * gm * x_hat;
                            }
                        }
                        float scale = inv_std / static_cast<float>(group_elems);
                        for (size_t ch = 0; ch < c_per_g; ++ch) {
                            size_t channel = c_start + ch;
                            size_t base = ((b * node.c) + channel) * node.spatial;
                            float gm = gamma->data[channel];
                            for (size_t s = 0; s < node.spatial; ++s) {
                                float x_hat = static_cast<float>(in_t->data[base + s] - mean) * inv_std;
                                float dy = grad_output[base + s];
                                gi[base + s] += scale * (static_cast<float>(group_elems) * dy * gm
                                    - sum_dy_gamma - x_hat * sum_dy_gamma_xhat);
                            }
                        }
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                accumulate_grad_map(gradients, node.inputs[1], gg);
                accumulate_grad_map(gradients, node.inputs[2], gb);
                break;
            }
            case Op::BatchNorm2d: {
                auto in_t = ctx.tensor(node.inputs[0]);
                auto gamma = ctx.tensor(node.inputs[1]);
                auto rmean = ctx.tensor(node.inputs[3]);
                auto rvar = ctx.tensor(node.inputs[4]);
                size_t spatial = node.n * node.h * node.w;
                std::vector<float> gi(node.n * node.c * node.h * node.w, 0.0f);
                std::vector<float> gg(node.c, 0.0f);
                std::vector<float> gb(node.c, 0.0f);
                for (size_t ch = 0; ch < node.c; ++ch) {
                    if (node.training) {
                        double mean = 0.0;
                        for (size_t b = 0; b < node.n; ++b)
                            for (size_t r = 0; r < node.h; ++r)
                                for (size_t col = 0; col < node.w; ++col)
                                    mean += in_t->data[((b * node.c + ch) * node.h + r) * node.w + col];
                        mean /= spatial;
                        double var = 0.0;
                        for (size_t b = 0; b < node.n; ++b)
                            for (size_t r = 0; r < node.h; ++r)
                                for (size_t col = 0; col < node.w; ++col) {
                                    double d = in_t->data[((b * node.c + ch) * node.h + r) * node.w + col] - mean;
                                    var += d * d;
                                }
                        var /= spatial;
                        double inv = 1.0 / std::sqrt(var + node.eps);
                        float sum_dy = 0.0f, sum_dy_yhat = 0.0f;
                        for (size_t b = 0; b < node.n; ++b)
                            for (size_t r = 0; r < node.h; ++r)
                                for (size_t col = 0; col < node.w; ++col) {
                                    size_t idx = ((b * node.c + ch) * node.h + r) * node.w + col;
                                    float norm = static_cast<float>((in_t->data[idx] - mean) * inv);
                                    float dy = grad_output[idx];
                                    sum_dy += dy;
                                    sum_dy_yhat += dy * norm;
                                    gg[ch] += dy * norm;
                                    gb[ch] += dy;
                                }
                        float scale = static_cast<float>(inv) / static_cast<float>(spatial);
                        for (size_t b = 0; b < node.n; ++b)
                            for (size_t r = 0; r < node.h; ++r)
                                for (size_t col = 0; col < node.w; ++col) {
                                    size_t idx = ((b * node.c + ch) * node.h + r) * node.w + col;
                                    float norm = static_cast<float>((in_t->data[idx] - mean) * inv);
                                    gi[idx] = scale * (static_cast<float>(spatial) * grad_output[idx] * gamma->data[ch]
                                                       - sum_dy * gamma->data[ch]
                                                       - norm * sum_dy_yhat * gamma->data[ch]);
                                }
                    } else {
                        float inv = static_cast<float>(1.0 / std::sqrt(rvar->data[ch] + node.eps));
                        for (size_t b = 0; b < node.n; ++b)
                            for (size_t r = 0; r < node.h; ++r)
                                for (size_t col = 0; col < node.w; ++col) {
                                    size_t idx = ((b * node.c + ch) * node.h + r) * node.w + col;
                                    gi[idx] = grad_output[idx] * gamma->data[ch] * inv;
                                    gg[ch] += grad_output[idx] * ((in_t->data[idx] - rmean->data[ch]) * inv);
                                    gb[ch] += grad_output[idx];
                                }
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                accumulate_grad_map(gradients, node.inputs[1], gg);
                accumulate_grad_map(gradients, node.inputs[2], gb);
                break;
            }
            case Op::Conv2d: {
                auto in_t = ctx.tensor(node.inputs[0]);
                auto weight = ctx.tensor(node.inputs[1]);
                size_t groups = std::max<size_t>(1, node.groups);
                size_t cin_per_g = node.c / groups;
                size_t cout_per_g = node.cout / groups;
                size_t kernel_len = cin_per_g * node.kh * node.kw;
                std::vector<float> gi(node.n * node.c * node.h * node.w, 0.0f);
                std::vector<float> gw(node.cout * kernel_len, 0.0f);
                for (size_t b = 0; b < node.n; ++b) {
                    for (size_t g = 0; g < groups; ++g) {
                        for (size_t oc = 0; oc < cout_per_g; ++oc) {
                            size_t cout_idx = g * cout_per_g + oc;
                            size_t w_base = cout_idx * kernel_len;
                            for (size_t orow = 0; orow < node.oh; ++orow) {
                                for (size_t ocol = 0; ocol < node.ow; ++ocol) {
                                    float go = grad_output[((b * node.cout + cout_idx) * node.oh + orow) * node.ow + ocol];
                                    for (size_t ic = 0; ic < cin_per_g; ++ic) {
                                        size_t cin_idx = g * cin_per_g + ic;
                                        size_t in_base = ((b * node.c + cin_idx) * node.h) * node.w;
                                        size_t w_ic_base = w_base + ic * node.kh * node.kw;
                                        for (size_t kr = 0; kr < node.kh; ++kr) {
                                            int64_t ir = static_cast<int64_t>(orow * node.sh + kr * node.dh) - static_cast<int64_t>(node.ph);
                                            if (ir < 0 || ir >= static_cast<int64_t>(node.h)) continue;
                                            for (size_t kc = 0; kc < node.kw; ++kc) {
                                                int64_t ic_col = static_cast<int64_t>(ocol * node.sw + kc * node.dw) - static_cast<int64_t>(node.pw);
                                                if (ic_col < 0 || ic_col >= static_cast<int64_t>(node.w)) continue;
                                                size_t in_idx = in_base + static_cast<size_t>(ir) * node.w + static_cast<size_t>(ic_col);
                                                size_t w_idx = w_ic_base + kr * node.kw + kc;
                                                gi[in_idx] += go * weight->data[w_idx];
                                                gw[w_idx] += go * in_t->data[in_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                accumulate_grad_map(gradients, node.inputs[1], gw);
                break;
            }
            case Op::MaxPool2d: {
                auto in_t = ctx.tensor(node.inputs[0]);
                std::vector<float> gi(node.n * node.c * node.h * node.w, 0.0f);
                for (size_t b = 0; b < node.n; ++b)
                    for (size_t ch = 0; ch < node.c; ++ch)
                        for (size_t orow = 0; orow < node.oh; ++orow)
                            for (size_t ocol = 0; ocol < node.ow; ++ocol) {
                                float mx = -std::numeric_limits<float>::infinity();
                                size_t mr = 0, mc = 0;
                                for (size_t kr = 0; kr < node.kh; ++kr) {
                                    size_t ir = orow * node.sh + kr;
                                    for (size_t kc = 0; kc < node.kw; ++kc) {
                                        size_t ic = ocol * node.sw + kc;
                                        float v = in_t->data[((b * node.c + ch) * node.h + ir) * node.w + ic];
                                        if (v > mx) { mx = v; mr = ir; mc = ic; }
                                    }
                                }
                                gi[((b * node.c + ch) * node.h + mr) * node.w + mc] +=
                                    grad_output[((b * node.c + ch) * node.oh + orow) * node.ow + ocol];
                            }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                break;
            }
            case Op::Dropout: {
                auto mask = dropout_mask(grad_output.size(), node.probability, node.training, node.seed);
                std::vector<float> gi(grad_output.size(), 0.0f);
                for (size_t i = 0; i < gi.size(); ++i) gi[i] = grad_output[i] * mask[i];
                accumulate_grad_map(gradients, node.inputs[0], gi);
                break;
            }
            case Op::Embedding: {
                auto indices = ctx.tensor(node.inputs[0]);
                auto weight = ctx.tensor(node.inputs[1]);
                std::vector<float> gw(weight->data.size(), 0.0f);
                for (size_t r = 0; r < node.rows; ++r) {
                    size_t idx = static_cast<size_t>(indices->data[r]);
                    for (size_t c = 0; c < node.dim; ++c) {
                        gw[idx * node.dim + c] += grad_output[r * node.dim + c];
                    }
                }
                accumulate_grad_map(gradients, node.inputs[1], gw);
                break;
            }
            case Op::BatchedMatMul: {
                auto left = ctx.tensor(node.inputs[0]);
                auto right = ctx.tensor(node.inputs[1]);
                std::vector<float> gl(node.batch * node.m * node.k, 0.0f);
                std::vector<float> gr(node.batch * node.k * node.n, 0.0f);
                for (size_t b = 0; b < node.batch; ++b) {
                    gemm_nt(grad_output.data() + b * node.m * node.n, right->data.data() + b * node.k * node.n,
                            gl.data() + b * node.m * node.k, node.m, node.k, node.n);
                    gemm_tn(left->data.data() + b * node.m * node.k, grad_output.data() + b * node.m * node.n,
                            gr.data() + b * node.k * node.n, node.m, node.k, node.n);
                }
                accumulate_grad_map(gradients, node.inputs[0], gl);
                accumulate_grad_map(gradients, node.inputs[1], gr);
                break;
            }
            case Op::Permute: {
                size_t rank = node.in_shape.size();
                std::vector<size_t> out_shape(rank);
                for (size_t i = 0; i < rank; ++i) out_shape[i] = node.in_shape[node.perm[i]];
                std::vector<size_t> in_strides(rank, 1);
                for (size_t i = rank - 1; i-- > 0;) in_strides[i] = in_strides[i + 1] * node.in_shape[i + 1];
                std::vector<float> gi(grad_output.size(), 0.0f);
                std::vector<size_t> coords(rank, 0);
                for (size_t oi = 0; oi < grad_output.size(); ++oi) {
                    size_t ii = 0;
                    for (size_t d = 0; d < rank; ++d) ii += coords[d] * in_strides[node.perm[d]];
                    gi[ii] += grad_output[oi];
                    for (size_t d = rank; d-- > 0;) {
                        if (++coords[d] < out_shape[d]) break;
                        coords[d] = 0;
                    }
                }
                accumulate_grad_map(gradients, node.inputs[0], gi);
                break;
            }
            case Op::Reshape: {
                accumulate_grad_map(gradients, node.inputs[0], grad_output);
                break;
            }
        }
    }

    for (auto& kv : gradients) {
        std::shared_ptr<NativeTensor> t;
        try {
            t = ctx.tensor(kv.first);
        } catch (...) {
            continue;
        }
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        if (t->requires_grad) {
            t->accumulate_grad(kv.second);
        }
    }
}

struct HistogramSession {
    std::vector<int8_t> bins;
    std::vector<double> gradients;
    std::vector<double> hessians;
    size_t feature_count;
    size_t max_bins;
};

static std::mutex SESSION_MTX;
static std::unordered_map<uint64_t, std::shared_ptr<HistogramSession>> SESSION_REGISTRY;

} // namespace jsl

using namespace jsl;

extern "C" {

// ======================= NativeBindings =======================

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_available(JNIEnv* env, jclass) {
    return run_jni<jint>(env, 0, []() { return 1; });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_version(JNIEnv* env, jclass) {
    return run_jni<jint>(env, 0, []() { return VERSION; });
}

JNIEXPORT jlong JNICALL Java_jsl_nativebridge_NativeBindings_createContext(JNIEnv* env, jclass) {
    return run_jni<jlong>(env, 0, []() {
        auto ctx = std::make_shared<Context>();
        uint64_t h = NEXT_CONTEXT_HANDLE.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(REGISTRY_MTX);
        CONTEXT_REGISTRY[h] = std::move(ctx);
        return static_cast<jlong>(h);
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_destroyContext(JNIEnv* env, jclass, jlong context) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        if (context == 0) return 0;
        uint64_t h = static_cast<uint64_t>(context);
        std::lock_guard<std::mutex> lk(REGISTRY_MTX);
        auto erased = CONTEXT_REGISTRY.erase(h);
        return erased > 0 ? 0 : 1;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_contextStats(JNIEnv* env, jclass, jlong context, jlongArray output) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        if (!output || env->GetArrayLength(output) < 3) {
            throw NativeError::invalid("contextStats expects a long[3] output array");
        }
        int64_t vals[3];
        ctx->stats(vals);
        jlong jvals[3] = {vals[0], vals[1], vals[2]};
        env->SetLongArrayRegion(output, 0, 3, jvals);
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_clearGraph(JNIEnv* env, jclass, jlong context) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        std::lock_guard<std::mutex> lk(ctx->graph_mtx);
        ctx->graph.clear();
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_graphSize(JNIEnv* env, jclass, jlong context) {
    return run_jni<jint>(env, 0, [&]() {
        auto ctx = context_from_ptr(context);
        std::lock_guard<std::mutex> lk(ctx->graph_mtx);
        return static_cast<jint>(ctx->graph.nodes.size());
    });
}

static std::vector<size_t> read_shape_jni(JNIEnv* env, jintArray shape_arr) {
    if (!shape_arr) throw NativeError::shape("tensor shape must not be null");
    jsize len = env->GetArrayLength(shape_arr);
    if (len <= 0) throw NativeError::shape("tensor shape must not be empty");
    std::vector<jint> raw(len);
    env->GetIntArrayRegion(shape_arr, 0, len, raw.data());
    std::vector<size_t> shape;
    shape.reserve(len);
    for (jint d : raw) {
        if (d <= 0) throw NativeError::shape("tensor dimensions must be positive");
        shape.push_back(static_cast<size_t>(d));
    }
    return shape;
}

JNIEXPORT jlong JNICALL Java_jsl_nativebridge_NativeBindings_createTensor(
    JNIEnv* env, jclass, jlong context, jintArray shape, jboolean requires_grad) {
    return run_jni<jlong>(env, 0, [&]() {
        auto ctx = context_from_ptr(context);
        auto s = read_shape_jni(env, shape);
        return static_cast<jlong>(ctx->create_tensor(std::move(s), requires_grad != 0));
    });
}

JNIEXPORT jlong JNICALL Java_jsl_nativebridge_NativeBindings_createTensorFromData(
    JNIEnv* env, jclass, jlong context, jfloatArray data, jintArray shape, jboolean requires_grad) {
    return run_jni<jlong>(env, 0, [&]() {
        auto ctx = context_from_ptr(context);
        auto s = read_shape_jni(env, shape);
        if (!data) throw NativeError::invalid("data array is null");
        jsize dlen = env->GetArrayLength(data);
        std::vector<float> vals(dlen);
        env->GetFloatArrayRegion(data, 0, dlen, vals.data());
        return static_cast<jlong>(ctx->create_tensor_from_data(std::move(s), std::move(vals), requires_grad != 0));
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_setTensorData(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloatArray data) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        jsize dlen = env->GetArrayLength(data);
        std::vector<float> vals(dlen);
        env->GetFloatArrayRegion(data, 0, dlen, vals.data());
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        t->set_data(vals);
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_tensorData(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloatArray output) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        jsize olen = env->GetArrayLength(output);
        if (static_cast<size_t>(olen) != t->data.size()) {
            throw NativeError::shape("output array length mismatch");
        }
        env->SetFloatArrayRegion(output, 0, olen, t->data.data());
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_tensorGrad(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloatArray output) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        jsize olen = env->GetArrayLength(output);
        if (static_cast<size_t>(olen) != t->data.size()) {
            throw NativeError::shape("output array length mismatch");
        }
        if (t->grad.empty()) {
            std::vector<float> zeros(t->data.size(), 0.0f);
            env->SetFloatArrayRegion(output, 0, olen, zeros.data());
        } else {
            env->SetFloatArrayRegion(output, 0, olen, t->grad.data());
        }
        return 0;
    });
}

JNIEXPORT jintArray JNICALL Java_jsl_nativebridge_NativeBindings_tensorShape(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jintArray>(env, nullptr, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        jintArray out = env->NewIntArray(static_cast<jsize>(t->shape.size()));
        std::vector<jint> dims(t->shape.size());
        for (size_t i = 0; i < t->shape.size(); ++i) dims[i] = static_cast<jint>(t->shape[i]);
        env->SetIntArrayRegion(out, 0, static_cast<jsize>(dims.size()), dims.data());
        return out;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_fillTensor(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloat value) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        if (!std::isfinite(value)) throw NativeError::invalid("fill value must be finite");
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        std::fill(t->data.begin(), t->data.end(), value);
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_randomizeTensor(
    JNIEnv* env, jclass, jlong context, jlong tensor, jlong seed, jfloat scale) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        if (!std::isfinite(scale) || scale < 0.0f) {
            throw NativeError::invalid("random scale must be finite and non-negative");
        }
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        uint64_t state = static_cast<uint64_t>(seed) | 1ULL;
        for (float& v : t->data) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            float unit = static_cast<float>(state >> 40) / static_cast<float>(1ULL << 24);
            v = (unit * 2.0f - 1.0f) * scale;
        }
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_zeroGrad(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        t->zero_grad();
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_releaseTensor(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        if (tensor == 0) return 1;
        return ctx->release_tensor(static_cast<uint64_t>(tensor)) ? 0 : 1;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_sgdStep(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloat lr, jfloat wd) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        if (!std::isfinite(lr) || lr < 0.0f) {
            throw NativeError::invalid("learning rate must be finite and non-negative");
        }
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        t->ensure_grad();
        for (size_t i = 0; i < t->data.size(); ++i) {
            t->data[i] -= lr * (t->grad[i] + wd * t->data[i]);
        }
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_adamStep(
    JNIEnv* env, jclass, jlong context, jlong tensor, jfloat lr, jfloat beta1, jfloat beta2,
    jfloat eps, jfloat wd, jlong step) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        if (!std::isfinite(lr) || lr < 0.0f) throw NativeError::invalid("learning rate must be finite and non-negative");
        if (beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f) throw NativeError::invalid("Adam betas must be in [0, 1)");
        if (!std::isfinite(eps) || eps <= 0.0f) throw NativeError::invalid("Adam epsilon must be positive");
        if (step <= 0) throw NativeError::invalid("Adam step must be positive");
        auto ctx = context_from_ptr(context);
        uint64_t h = static_cast<uint64_t>(tensor);
        auto t = ctx->tensor(h);
        std::unique_lock<std::shared_mutex> lk(t->mtx);
        t->ensure_grad();
        size_t len = t->data.size();
        std::lock_guard<std::mutex> olk(ctx->opt_mtx);
        auto& st = ctx->optimizers[h];
        if (st.first.size() != len) {
            st.first.assign(len, 0.0f);
            st.second.assign(len, 0.0f);
        }
        float b1_corr = 1.0f - std::pow(beta1, static_cast<int>(step));
        float b2_corr = 1.0f - std::pow(beta2, static_cast<int>(step));
        for (size_t i = 0; i < len; ++i) {
            float g = t->grad[i];
            st.first[i] = beta1 * st.first[i] + (1.0f - beta1) * g;
            st.second[i] = beta2 * st.second[i] + (1.0f - beta2) * g * g;
            float m_hat = st.first[i] / b1_corr;
            float v_hat = st.second[i] / b2_corr;
            t->data[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * t->data[i]);
        }
        return 0;
    });
}

JNIEXPORT jdouble JNICALL Java_jsl_nativebridge_NativeBindings_clipGradNorm(
    JNIEnv* env, jclass, jlong context, jlongArray tensors, jfloat max_norm) {
    return run_jni<jdouble>(env, 0.0, [&]() {
        if (!std::isfinite(max_norm) || max_norm < 0.0f) {
            throw NativeError::invalid("gradient clipping norm must be finite and non-negative");
        }
        auto ctx = context_from_ptr(context);
        jsize len = env->GetArrayLength(tensors);
        std::vector<jlong> handles(len);
        env->GetLongArrayRegion(tensors, 0, len, handles.data());
        std::vector<std::shared_ptr<NativeTensor>> ptrs;
        ptrs.reserve(len);
        double total = 0.0;
        for (jlong h : handles) {
            auto t = ctx->tensor(static_cast<uint64_t>(h));
            for (float g : t->grad) total += static_cast<double>(g) * static_cast<double>(g);
            ptrs.push_back(t);
        }
        float norm = static_cast<float>(std::sqrt(total));
        if (norm > max_norm && norm > 0.0f) {
            float scale = max_norm / (norm + 1e-6f);
            for (auto& t : ptrs) {
                for (float& g : t->grad) g *= scale;
            }
        }
        return static_cast<jdouble>(norm);
    });
}

JNIEXPORT jdouble JNICALL Java_jsl_nativebridge_NativeBindings_tensorNorm(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jdouble>(env, 0.0, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        double sum = 0.0;
        for (float v : t->data) sum += static_cast<double>(v) * static_cast<double>(v);
        return std::sqrt(sum);
    });
}

JNIEXPORT jdouble JNICALL Java_jsl_nativebridge_NativeBindings_tensorGradNorm(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jdouble>(env, 0.0, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        double sum = 0.0;
        for (float v : t->grad) sum += static_cast<double>(v) * static_cast<double>(v);
        return std::sqrt(sum);
    });
}

JNIEXPORT jboolean JNICALL Java_jsl_nativebridge_NativeBindings_tensorIsFinite(
    JNIEnv* env, jclass, jlong context, jlong tensor) {
    return run_jni<jboolean>(env, 0, [&]() {
        auto ctx = context_from_ptr(context);
        auto t = ctx->tensor(static_cast<uint64_t>(tensor));
        std::shared_lock<std::shared_mutex> lk(t->mtx);
        for (float v : t->data) {
            if (!std::isfinite(v)) return static_cast<jboolean>(0);
        }
        return static_cast<jboolean>(1);
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_executeQueue(
    JNIEnv* env, jclass, jlong context, jobject commands, jint count, jint /*threads*/) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        if (count < 0) throw NativeError::invalid("command count must be non-negative");
        if (count == 0) return 0;
        const uint8_t* addr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(commands));
        jlong cap = env->GetDirectBufferCapacity(commands);
        size_t req = static_cast<size_t>(count) * sizeof(OpCommand);
        if (!addr || cap < static_cast<jlong>(req)) {
            throw NativeError::invalid("direct command buffer capacity insufficient or null");
        }
        std::lock_guard<std::mutex> glk(ctx->graph_mtx);
        for (int i = 0; i < count; ++i) {
            OpCommand cmd;
            std::memcpy(&cmd, addr + i * sizeof(OpCommand), sizeof(OpCommand));
            execute_command(*ctx, ctx->graph, cmd);
        }
        return 0;
    });
}

JNIEXPORT jint JNICALL Java_jsl_nativebridge_NativeBindings_backward(
    JNIEnv* env, jclass, jlong context, jlong root, jint /*threads*/) {
    return run_jni<jint>(env, STATUS_INVALID, [&]() {
        auto ctx = context_from_ptr(context);
        backward_impl(*ctx, static_cast<uint64_t>(root));
        return 0;
    });
}

// ======================= NativeKernels =======================

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_version(JNIEnv*, jclass) {
    return 1;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_gemm(
    JNIEnv* env, jclass, jdoubleArray a_arr, jdoubleArray b_arr, jdoubleArray c_arr,
    jint m, jint k, jint n, jint /*threads*/) {
    if (m <= 0 || k <= 0 || n <= 0) return STATUS_INVALID;
    size_t M = m, K = k, N = n;
    if (static_cast<size_t>(env->GetArrayLength(a_arr)) != M * K ||
        static_cast<size_t>(env->GetArrayLength(b_arr)) != K * N ||
        static_cast<size_t>(env->GetArrayLength(c_arr)) != M * N) {
        return STATUS_INVALID;
    }
    std::vector<double> a(M * K), b(K * N), c(M * N, 0.0);
    env->GetDoubleArrayRegion(a_arr, 0, static_cast<jsize>(M * K), a.data());
    env->GetDoubleArrayRegion(b_arr, 0, static_cast<jsize>(K * N), b.data());
    for (size_t i = 0; i < M; ++i) {
        double* c_row = c.data() + i * N;
        const double* a_row = a.data() + i * K;
        for (size_t p = 0; p < K; ++p) {
            double alpha = a_row[p];
            const double* b_row = b.data() + p * N;
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += alpha * b_row[j];
            }
        }
    }
    env->SetDoubleArrayRegion(c_arr, 0, static_cast<jsize>(M * N), c.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_kmeansAssign(
    JNIEnv* env, jclass, jdoubleArray data_arr, jdoubleArray centers_arr,
    jint rows, jint clusters, jint features, jint /*threads*/,
    jintArray assign_arr, jdoubleArray dist_arr) {
    if (rows <= 0 || clusters <= 0 || features <= 0) return STATUS_INVALID;
    size_t R = rows, C = clusters, F = features;
    std::vector<double> data(R * F), centers(C * F), dist(R);
    std::vector<jint> assign(R);
    env->GetDoubleArrayRegion(data_arr, 0, static_cast<jsize>(R * F), data.data());
    env->GetDoubleArrayRegion(centers_arr, 0, static_cast<jsize>(C * F), centers.data());
    for (size_t r = 0; r < R; ++r) {
        const double* pt = data.data() + r * F;
        size_t best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < C; ++c) {
            const double* ctr = centers.data() + c * F;
            double sum = 0.0;
            for (size_t f = 0; f < F; ++f) {
                double d = pt[f] - ctr[f];
                sum += d * d;
            }
            if (sum < best_d) { best_d = sum; best = c; }
        }
        assign[r] = static_cast<jint>(best);
        dist[r] = best_d;
    }
    env->SetIntArrayRegion(assign_arr, 0, rows, assign.data());
    env->SetDoubleArrayRegion(dist_arr, 0, rows, dist.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_forestPredict(
    JNIEnv* env, jclass, jdoubleArray data_arr, jint rows, jint cols,
    jintArray feat_arr, jdoubleArray thresh_arr, jintArray left_arr, jintArray right_arr,
    jdoubleArray val_arr, jintArray off_arr, jdoubleArray out_arr, jdouble scale) {
    if (rows <= 0 || cols <= 0) return STATUS_INVALID;
    jsize nodes = env->GetArrayLength(feat_arr);
    jsize offs = env->GetArrayLength(off_arr);
    if (offs < 2) return STATUS_INVALID;
    std::vector<double> data(rows * cols), thresh(nodes), val(nodes), out(rows);
    std::vector<jint> feat(nodes), left(nodes), right(nodes), offsets(offs);
    env->GetDoubleArrayRegion(data_arr, 0, rows * cols, data.data());
    env->GetIntArrayRegion(feat_arr, 0, nodes, feat.data());
    env->GetDoubleArrayRegion(thresh_arr, 0, nodes, thresh.data());
    env->GetIntArrayRegion(left_arr, 0, nodes, left.data());
    env->GetIntArrayRegion(right_arr, 0, nodes, right.data());
    env->GetDoubleArrayRegion(val_arr, 0, nodes, val.data());
    env->GetIntArrayRegion(off_arr, 0, offs, offsets.data());
    env->GetDoubleArrayRegion(out_arr, 0, rows, out.data());
    int trees = offs - 1;
    for (int r = 0; r < rows; ++r) {
        const double* row_ptr = data.data() + r * cols;
        double sum = 0.0;
        for (int t = 0; t < trees; ++t) {
            int node = offsets[t];
            while (true) {
                int sf = feat[node];
                if (sf < 0) { sum += scale * val[node]; break; }
                node = (row_ptr[sf] <= thresh[node]) ? left[node] : right[node];
            }
        }
        out[r] += sum;
    }
    env->SetDoubleArrayRegion(out_arr, 0, rows, out.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_forestVote(
    JNIEnv* env, jclass, jdoubleArray data_arr, jint rows, jint cols,
    jintArray feat_arr, jdoubleArray thresh_arr, jintArray left_arr, jintArray right_arr,
    jdoubleArray val_arr, jintArray off_arr, jdoubleArray labels_arr, jdoubleArray votes_arr, jdouble scale) {
    if (rows <= 0 || cols <= 0) return STATUS_INVALID;
    jsize nodes = env->GetArrayLength(feat_arr);
    jsize offs = env->GetArrayLength(off_arr);
    jsize classes = env->GetArrayLength(labels_arr);
    if (offs < 2 || classes <= 0) return STATUS_INVALID;
    std::vector<double> data(rows * cols), thresh(nodes), val(nodes), labels(classes), votes(rows * classes);
    std::vector<jint> feat(nodes), left(nodes), right(nodes), offsets(offs);
    env->GetDoubleArrayRegion(data_arr, 0, rows * cols, data.data());
    env->GetIntArrayRegion(feat_arr, 0, nodes, feat.data());
    env->GetDoubleArrayRegion(thresh_arr, 0, nodes, thresh.data());
    env->GetIntArrayRegion(left_arr, 0, nodes, left.data());
    env->GetIntArrayRegion(right_arr, 0, nodes, right.data());
    env->GetDoubleArrayRegion(val_arr, 0, nodes, val.data());
    env->GetIntArrayRegion(off_arr, 0, offs, offsets.data());
    env->GetDoubleArrayRegion(labels_arr, 0, classes, labels.data());
    env->GetDoubleArrayRegion(votes_arr, 0, rows * classes, votes.data());
    int trees = offs - 1;
    for (int t = 0; t < trees; ++t) {
        int base = offsets[t];
        for (int r = 0; r < rows; ++r) {
            const double* row_ptr = data.data() + r * cols;
            int node = base;
            while (true) {
                int sf = feat[node];
                if (sf < 0) {
                    double lbl = val[node];
                    for (int c = 0; c < classes; ++c) {
                        if (labels[c] == lbl) { votes[r * classes + c] += scale; break; }
                    }
                    break;
                }
                node = (row_ptr[sf] <= thresh[node]) ? left[node] : right[node];
            }
        }
    }
    env->SetDoubleArrayRegion(votes_arr, 0, rows * classes, votes.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_histogramAccumulate(
    JNIEnv* env, jclass, jdoubleArray grad_arr, jdoubleArray hess_arr, jbyteArray bins_arr,
    jintArray samp_arr, jintArray active_arr, jint feature_count, jint max_bins, jint /*threads*/,
    jdoubleArray out_g_arr, jdoubleArray out_h_arr, jintArray out_c_arr) {
    if (feature_count <= 0 || max_bins <= 0) return STATUS_INVALID;
    jsize n_samp = env->GetArrayLength(samp_arr);
    jsize n_act = env->GetArrayLength(active_arr);
    jsize total = feature_count * max_bins;
    if (n_act <= 0) return STATUS_INVALID;
    jsize n_rows = env->GetArrayLength(grad_arr);
    std::vector<double> grad(n_rows), hess(n_rows), out_g(total), out_h(total);
    std::vector<jbyte> bins(env->GetArrayLength(bins_arr));
    std::vector<jint> samp(n_samp), active(n_act), out_c(total);
    env->GetDoubleArrayRegion(grad_arr, 0, n_rows, grad.data());
    env->GetDoubleArrayRegion(hess_arr, 0, n_rows, hess.data());
    env->GetByteArrayRegion(bins_arr, 0, static_cast<jsize>(bins.size()), bins.data());
    env->GetIntArrayRegion(samp_arr, 0, n_samp, samp.data());
    env->GetIntArrayRegion(active_arr, 0, n_act, active.data());
    env->GetDoubleArrayRegion(out_g_arr, 0, total, out_g.data());
    env->GetDoubleArrayRegion(out_h_arr, 0, total, out_h.data());
    env->GetIntArrayRegion(out_c_arr, 0, total, out_c.data());
    for (jsize i = 0; i < n_samp; ++i) {
        size_t s = static_cast<size_t>(samp[i]);
        size_t row_base = s * feature_count;
        double g = grad[s], h = hess[s];
        for (jsize a = 0; a < n_act; ++a) {
            size_t f = static_cast<size_t>(active[a]);
            size_t bucket = f * max_bins + static_cast<uint8_t>(bins[row_base + f]);
            out_g[bucket] += g;
            out_h[bucket] += h;
            out_c[bucket] += 1;
        }
    }
    env->SetDoubleArrayRegion(out_g_arr, 0, total, out_g.data());
    env->SetDoubleArrayRegion(out_h_arr, 0, total, out_h.data());
    env->SetIntArrayRegion(out_c_arr, 0, total, out_c.data());
    return STATUS_OK;
}

JNIEXPORT jlong JNICALL Java_jsl_core_NativeKernels_histogramSessionCreate(
    JNIEnv* env, jclass, jbyteArray bins_arr, jdoubleArray grad_arr, jdoubleArray hess_arr,
    jint feature_count, jint max_bins) {
    if (feature_count <= 0 || max_bins <= 0) return 0;
    jsize blen = env->GetArrayLength(bins_arr);
    jsize glen = env->GetArrayLength(grad_arr);
    jsize hlen = env->GetArrayLength(hess_arr);
    if (blen % feature_count != 0 || glen != hlen) return 0;
    auto sess = std::make_shared<HistogramSession>();
    sess->feature_count = static_cast<size_t>(feature_count);
    sess->max_bins = static_cast<size_t>(max_bins);
    sess->bins.resize(blen);
    sess->gradients.resize(glen);
    sess->hessians.resize(hlen);
    env->GetByteArrayRegion(bins_arr, 0, blen, reinterpret_cast<jbyte*>(sess->bins.data()));
    env->GetDoubleArrayRegion(grad_arr, 0, glen, sess->gradients.data());
    env->GetDoubleArrayRegion(hess_arr, 0, hlen, sess->hessians.data());
    uint64_t h = NEXT_SESSION_HANDLE.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(SESSION_MTX);
    SESSION_REGISTRY[h] = std::move(sess);
    return static_cast<jlong>(h);
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_histogramSessionUpdate(
    JNIEnv* env, jclass, jlong session, jdoubleArray grad_arr, jdoubleArray hess_arr) {
    std::shared_ptr<HistogramSession> sess;
    {
        std::lock_guard<std::mutex> lk(SESSION_MTX);
        auto it = SESSION_REGISTRY.find(static_cast<uint64_t>(session));
        if (it == SESSION_REGISTRY.end()) return STATUS_INVALID;
        sess = it->second;
    }
    jsize glen = env->GetArrayLength(grad_arr);
    jsize hlen = env->GetArrayLength(hess_arr);
    if (static_cast<size_t>(glen) != sess->gradients.size() || static_cast<size_t>(hlen) != sess->hessians.size()) {
        return STATUS_INVALID;
    }
    env->GetDoubleArrayRegion(grad_arr, 0, glen, sess->gradients.data());
    env->GetDoubleArrayRegion(hess_arr, 0, hlen, sess->hessians.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_histogramSessionAccumulate(
    JNIEnv* env, jclass, jlong session, jintArray samp_arr, jintArray active_arr, jint /*threads*/,
    jdoubleArray out_g_arr, jdoubleArray out_h_arr, jintArray out_c_arr) {
    std::shared_ptr<HistogramSession> sess;
    {
        std::lock_guard<std::mutex> lk(SESSION_MTX);
        auto it = SESSION_REGISTRY.find(static_cast<uint64_t>(session));
        if (it == SESSION_REGISTRY.end()) return STATUS_INVALID;
        sess = it->second;
    }
    jsize n_samp = env->GetArrayLength(samp_arr);
    jsize n_act = env->GetArrayLength(active_arr);
    size_t total = sess->feature_count * sess->max_bins;
    if (n_act <= 0) return STATUS_INVALID;
    std::vector<jint> samp(n_samp), active(n_act), out_c(total);
    std::vector<double> out_g(total), out_h(total);
    env->GetIntArrayRegion(samp_arr, 0, n_samp, samp.data());
    env->GetIntArrayRegion(active_arr, 0, n_act, active.data());
    env->GetDoubleArrayRegion(out_g_arr, 0, static_cast<jsize>(total), out_g.data());
    env->GetDoubleArrayRegion(out_h_arr, 0, static_cast<jsize>(total), out_h.data());
    env->GetIntArrayRegion(out_c_arr, 0, static_cast<jsize>(total), out_c.data());
    for (jsize i = 0; i < n_samp; ++i) {
        size_t s = static_cast<size_t>(samp[i]);
        size_t row_base = s * sess->feature_count;
        double g = sess->gradients[s], h = sess->hessians[s];
        for (jsize a = 0; a < n_act; ++a) {
            size_t f = static_cast<size_t>(active[a]);
            size_t bucket = f * sess->max_bins + static_cast<uint8_t>(sess->bins[row_base + f]);
            out_g[bucket] += g;
            out_h[bucket] += h;
            out_c[bucket] += 1;
        }
    }
    env->SetDoubleArrayRegion(out_g_arr, 0, static_cast<jsize>(total), out_g.data());
    env->SetDoubleArrayRegion(out_h_arr, 0, static_cast<jsize>(total), out_h.data());
    env->SetIntArrayRegion(out_c_arr, 0, static_cast<jsize>(total), out_c.data());
    return STATUS_OK;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_histogramSessionDestroy(
    JNIEnv*, jclass, jlong session) {
    if (session == 0) return 0;
    std::lock_guard<std::mutex> lk(SESSION_MTX);
    return SESSION_REGISTRY.erase(static_cast<uint64_t>(session)) > 0 ? 0 : 1;
}

// Return STATUS_INVALID on optional fastmath entry points so Java uses its exact reference path
// or execute them directly:
JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_ridgeFit(
    JNIEnv*, jclass, jdoubleArray, jint, jint, jdoubleArray, jdouble, jboolean, jint, jdoubleArray, jdoubleArray) {
    return STATUS_INVALID;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_gaussianNbFit(
    JNIEnv*, jclass, jdoubleArray, jint, jint, jintArray, jint, jdoubleArray, jdoubleArray, jdoubleArray, jintArray, jdouble) {
    return STATUS_INVALID;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_kmeansFit(
    JNIEnv*, jclass, jdoubleArray, jint, jint, jint, jint, jint, jdouble, jlong, jint, jdoubleArray, jdoubleArray, jintArray) {
    return STATUS_INVALID;
}

JNIEXPORT jint JNICALL Java_jsl_core_NativeKernels_binFeatures(
    JNIEnv*, jclass, jdoubleArray, jint, jint, jint, jbyteArray, jdoubleArray, jintArray) {
    return STATUS_INVALID;
}

} // extern "C"
