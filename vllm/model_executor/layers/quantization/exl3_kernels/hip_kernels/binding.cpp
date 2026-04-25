#include <torch/extension.h>

void hip_rms_norm(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    bool span_heads);

void hip_silu_mul(at::Tensor g, at::Tensor u, at::Tensor z);
void hip_gelu_mul(at::Tensor g, at::Tensor u, at::Tensor z);
void hip_relu2_mul(at::Tensor g, at::Tensor u, at::Tensor z);

void hip_had_r_128(
    at::Tensor x,
    at::Tensor out,
    c10::optional<at::Tensor> pre_scale,
    c10::optional<at::Tensor> post_scale,
    float r_scale);

void hip_batched_had_r_128(
    at::Tensor x_sorted,
    at::Tensor scale_stacked,
    at::Tensor output,
    at::Tensor expert_ids,
    int pre);

void hip_dequant_cache_paged(
    at::Tensor qk,
    at::Tensor sk,
    at::Tensor k_out,
    at::Tensor qv,
    at::Tensor sv,
    at::Tensor v_out,
    at::Tensor cache_seqlens,
    at::Tensor block_table,
    int page_size);

void hip_quant_cache_paged(
    at::Tensor k_in,
    at::Tensor qk,
    at::Tensor sk,
    at::Tensor v_in,
    at::Tensor qv,
    at::Tensor sv,
    at::Tensor cache_seqlens,
    at::Tensor block_table,
    int page_size,
    int length);

void hip_exl3_gemm(
    at::Tensor A,
    at::Tensor B_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int bits,
    int cb);

void hip_exl3_gemm_v2(
    at::Tensor A,
    at::Tensor B_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial);

void hip_exl3_fused_moe_gemm(
    at::Tensor A,
    at::Tensor B_stacked_i32,
    at::Tensor C,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int EM_max,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial);

void hip_exl3_fused_moe_gemm_m64(
    at::Tensor A,
    at::Tensor B_stacked_i32,
    at::Tensor C,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int EM_max,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial);

void hip_exl3_fused_moe_gemm_fp16(
    at::Tensor A,
    at::Tensor B_stacked_fp16,
    at::Tensor C,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    int EM_max,
    int split_k,
    at::Tensor C_partial);

void hip_exl3_gemm_v3(
    at::Tensor A,
    at::Tensor B_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    at::Tensor locks,
    int bits,
    int cb,
    int split_k);

void hip_exl3_fused_moe_gemm_v3(
    at::Tensor A,
    at::Tensor B_stacked_i32,
    at::Tensor C,
    at::Tensor expert_ids,
    at::Tensor num_tokens_post_padded,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    at::Tensor locks,
    int EM_max,
    int bits,
    int cb,
    int split_k);

void hip_exl3_gemm_v4(
    at::Tensor A,
    at::Tensor B_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial);

void hip_exl3_batched_gemm_v4(
    at::Tensor A_batched,
    at::Tensor B_stacked_i32,
    at::Tensor C,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int num_outputs,
    int bits,
    int cb,
    int split_k,
    at::Tensor C_partial);

void hip_rocwmma_bench(
    at::Tensor A,
    at::Tensor B_i32,
    at::Tensor word_idx,
    at::Tensor next_word_idx,
    at::Tensor shift_tbl,
    int bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm", &hip_rms_norm,
          "Fused RMSNorm for ROCm (HIP)",
          py::arg("x"),
          py::arg("w"),
          py::arg("y"),
          py::arg("epsilon"),
          py::arg("constant_bias") = 0.0f,
          py::arg("span_heads") = false);

    m.def("silu_mul", &hip_silu_mul,
          "Fused SiLU(g) * u -> z (HIP)",
          py::arg("g"), py::arg("u"), py::arg("z"));

    m.def("gelu_mul", &hip_gelu_mul,
          "Fused GELU(g) * u -> z (HIP)",
          py::arg("g"), py::arg("u"), py::arg("z"));

    m.def("relu2_mul", &hip_relu2_mul,
          "Fused ReLU(g)^2 * u -> z (HIP)",
          py::arg("g"), py::arg("u"), py::arg("z"));

    m.def("had_r_128", &hip_had_r_128,
          "Butterfly Hadamard-128 transform (HIP)",
          py::arg("x"),
          py::arg("out"),
          py::arg("pre_scale"),
          py::arg("post_scale"),
          py::arg("r_scale"));

    m.def("batched_had_r_128", &hip_batched_had_r_128,
          "Batched DPP-fused Hadamard-128 for MoE (HIP)",
          py::arg("x_sorted"),
          py::arg("scale_stacked"),
          py::arg("output"),
          py::arg("expert_ids"),
          py::arg("pre"));

    m.def("dequant_cache_paged", &hip_dequant_cache_paged,
          "Dequantize paged KV cache (HIP)",
          py::arg("qk"),
          py::arg("sk"),
          py::arg("k_out"),
          py::arg("qv"),
          py::arg("sv"),
          py::arg("v_out"),
          py::arg("cache_seqlens"),
          py::arg("block_table"),
          py::arg("page_size"));

    m.def("quant_cache_paged", &hip_quant_cache_paged,
          "Quantize into paged KV cache (HIP)",
          py::arg("k_in"),
          py::arg("qk"),
          py::arg("sk"),
          py::arg("v_in"),
          py::arg("qv"),
          py::arg("sv"),
          py::arg("cache_seqlens"),
          py::arg("block_table"),
          py::arg("page_size"),
          py::arg("length"));

    m.def("exl3_gemm", &hip_exl3_gemm,
          "EXL3 dequant + GEMM using rocWMMA (HIP)",
          py::arg("A"),
          py::arg("B_i32"),
          py::arg("C"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("bits"),
          py::arg("cb"));

    m.def("exl3_gemm_v2", &hip_exl3_gemm_v2,
          "EXL3 dequant + GEMM with split-K + direct A load (HIP Phase 2)",
          py::arg("A"),
          py::arg("B_i32"),
          py::arg("C"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("exl3_fused_moe_gemm", &hip_exl3_fused_moe_gemm,
          "Fused MoE EXL3 dequant + GEMM using rocWMMA (HIP Phase 3)",
          py::arg("A"),
          py::arg("B_stacked_i32"),
          py::arg("C"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("EM_max"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("exl3_fused_moe_gemm_m64", &hip_exl3_fused_moe_gemm_m64,
          "Fused MoE EXL3 dequant + GEMM with BLOCK_M=64 for prefill (HIP Phase 3d)",
          py::arg("A"),
          py::arg("B_stacked_i32"),
          py::arg("C"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("EM_max"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("exl3_fused_moe_gemm_fp16", &hip_exl3_fused_moe_gemm_fp16,
          "Fused MoE FP16 GEMM using rocWMMA (HIP Phase 4)",
          py::arg("A"),
          py::arg("B_stacked_fp16"),
          py::arg("C"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          py::arg("EM_max"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("exl3_gemm_v3", &hip_exl3_gemm_v3,
          "EXL3 pipelined 4-wave dequant+GEMM with lock-based split-K (HIP v3)",
          py::arg("A"),
          py::arg("B_i32"),
          py::arg("C"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("locks"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"));

    m.def("exl3_fused_moe_gemm_v3", &hip_exl3_fused_moe_gemm_v3,
          "Fused MoE EXL3 pipelined 4-wave dequant+GEMM with lock-based split-K (HIP v3)",
          py::arg("A"),
          py::arg("B_stacked_i32"),
          py::arg("C"),
          py::arg("expert_ids"),
          py::arg("num_tokens_post_padded"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("locks"),
          py::arg("EM_max"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"));

    m.def("exl3_gemm_v4", &hip_exl3_gemm_v4,
          "EXL3 register-only VALU dequant+GEMM for M=1 decode (HIP V4)",
          py::arg("A"),
          py::arg("B_i32"),
          py::arg("C"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("exl3_batched_gemm_v4", &hip_exl3_batched_gemm_v4,
          "Batched EXL3 V4 VALU dequant+GEMM for M=1 multi-GEMM (HIP)",
          py::arg("A_batched"),
          py::arg("B_stacked_i32"),
          py::arg("C"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("num_outputs"),
          py::arg("bits"),
          py::arg("cb"),
          py::arg("split_k"),
          py::arg("C_partial"));

    m.def("rocwmma_bench", &hip_rocwmma_bench,
          "rocWMMA microbenchmark — runs all operations (HIP)",
          py::arg("A"),
          py::arg("B_i32"),
          py::arg("word_idx"),
          py::arg("next_word_idx"),
          py::arg("shift_tbl"),
          py::arg("bits"));
}
