/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <mutex>
#include <tuple>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include "ck_fmha_util.h"
#include "ck_tiled_fmha_fwd_splitkv_selector.h"
#include "ck_tiled_fmha_params.h"

extern void batched_forward_fp16(
    BatchedForwardParams& param,
    hipStream_t stream);
extern void batched_forward_bf16(
    BatchedForwardParams& param,
    hipStream_t stream);
extern void grouped_forward_fp16(
    GroupedForwardParams& param,
    hipStream_t stream);
extern void grouped_forward_bf16(
    GroupedForwardParams& param,
    hipStream_t stream);

extern void batched_infer_fp16(BatchedForwardParams& param, hipStream_t stream);
extern void batched_infer_bf16(BatchedForwardParams& param, hipStream_t stream);
extern void grouped_infer_fp16(GroupedForwardParams& param, hipStream_t stream);
extern void grouped_infer_bf16(GroupedForwardParams& param, hipStream_t stream);

namespace {

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<at::Tensor, std::optional<at::Tensor>, int64_t, int64_t>
efficient_attention_forward_ck(
    const at::Tensor& query, // [b, seqlen, num_heads_q, K]
    const at::Tensor& key, // [b, seqlen, num_heads_kv, K]
    const at::Tensor& value, // [b, seqlen, num_heads_kv, Kv]
    const c10::optional<at::Tensor>& bias, // [b, num_heads_q, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    double dropout_p, // attention matrix dropout probability
    bool compute_logsumexp,
    int64_t custom_mask_type,
    c10::optional<double> scale,
    const c10::optional<at::Tensor>& seqlen_k,
    const c10::optional<int64_t> window_size,
    const c10::optional<at::Tensor>& block_tables,
    const c10::optional<int64_t> page_size) {
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) % key.size(2) == 0);
  TORCH_CHECK(key.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  TORCH_CHECK(query.scalar_type() == key.scalar_type());
  TORCH_CHECK(query.scalar_type() == value.scalar_type());

  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    TORCH_CHECK(
        seqstart_q->size(0) == seqstart_k->size(0) ||
        seqstart_q->size(0) == seqstart_k->size(0) + 1);
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
  };

  TORCH_CHECK(block_tables.has_value() == page_size.has_value());
  TORCH_CHECK(!block_tables.has_value() || block_tables->dim() == 2);

  // Currently xformers only use Paged-KVcache in grouped mode
  TORCH_CHECK(seqstart_q.has_value() || !block_tables.has_value());

  // last dim is contiguous, device is kCUDA
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t Hq = query.size(-2);
  int64_t Hkv = key.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  auto opts = query.options();

  std::optional<at::Tensor> logsumexp = std::nullopt;

  at::Tensor out = at::empty({B, M, Hq, Kv}, opts);

  at::Tensor logsumexp_acc;
  at::Tensor out_acc;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  int64_t philox_seed = 0;
  int64_t philox_offset = 0;

  if (use_dropout) {
    at::PhiloxCudaState rng_engine_inputs;
    at::CUDAGeneratorImpl* gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    rng_engine_inputs =
        gen->philox_cuda_state((B + 3) * (Hq + 1) * (M + 1) * (N + 1));

    const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);

    philox_seed = std::get<0>(seeds);
    philox_offset = std::get<1>(seeds);
  }

  auto set_batched_forward_params = [&](BatchedForwardParams& p) {
    p.B = B;
    p.M = M;
    p.N = N;
    p.Hq = Hq;
    p.Hkv = Hkv;
    p.K = K;
    p.Kv = Kv;

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(K)));
    }

    p.q_ptr = query.data_ptr();
    p.k_ptr = key.data_ptr();
    p.v_ptr = value.data_ptr();
    p.out_ptr = out.data_ptr();

    p.q_strides = {
        static_cast<int>(query.stride(0)),
        static_cast<int>(query.stride(1)),
        static_cast<int>(query.stride(2)),
        static_cast<int>(query.stride(3))};
    p.k_strides = {
        static_cast<int>(key.stride(0)),
        static_cast<int>(key.stride(1)),
        static_cast<int>(key.stride(2)),
        static_cast<int>(key.stride(3))};
    p.v_strides = {
        static_cast<int>(value.stride(0)),
        static_cast<int>(value.stride(1)),
        static_cast<int>(value.stride(2)),
        static_cast<int>(value.stride(3))};
    p.out_strides = {
        static_cast<int>(out.stride(0)),
        static_cast<int>(out.stride(1)),
        static_cast<int>(out.stride(2)),
        static_cast<int>(out.stride(3))};

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      p.attn_bias_ptr = bias->data_ptr();

      const at::Tensor bias_4d_view = get_bias_4d_view(*bias, B, Hq, M, N);
      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};
    } else
      p.has_attn_bias = false;

    p.custom_mask_type = custom_mask_type;
    p.window_size =
        window_size.has_value() ? (*window_size > 0 ? *window_size : 0) : 0;

    p.philox_seed = philox_seed;
    p.philox_offset = philox_offset;
    p.compute_logsumexp = compute_logsumexp;

    // the following parameters are only used by training forward
    if (use_dropout) {
      p.dropout_prob = static_cast<float>(dropout_p);
    } else
      p.dropout_prob = 0.0f;

    if (p.compute_logsumexp) {
      logsumexp = at::empty({B, Hq, M}, opts.dtype(at::kFloat));
      p.logsumexp_ptr = logsumexp->data_ptr();
      p.lse_strides = {
          static_cast<int>(logsumexp->stride(0)),
          static_cast<int>(logsumexp->stride(1)),
          static_cast<int>(logsumexp->stride(2))};
    } else {
      p.logsumexp_ptr = nullptr;
      p.lse_strides = {0, 0, 0};
    }

    bool use_split_kv;
    int num_kv_splits;

    std::tie(use_split_kv, num_kv_splits) =
        get_num_kv_splits_heuristic(p.B, p.Hq, p.M, std::max(p.K, p.Kv), 8);

    // 1) fmha fwd split-kv kernel does not support dropout
    p.use_split_kv = (!use_dropout && use_split_kv) ? true : false;

    p.num_kv_splits = num_kv_splits;

    if (p.use_split_kv && p.num_kv_splits > 1) {
      out_acc =
          at::empty({p.num_kv_splits, B, M, Hq, Kv}, opts.dtype(at::kFloat));
      p.out_acc_ptr = out_acc.data_ptr();
      p.out_acc_strides = {
          static_cast<int>(out_acc.stride(0)),
          static_cast<int>(out_acc.stride(1)),
          static_cast<int>(out_acc.stride(2)),
          static_cast<int>(out_acc.stride(3)),
          static_cast<int>(out_acc.stride(4))};

      logsumexp_acc =
          at::empty({p.num_kv_splits, B, Hq, M}, opts.dtype(at::kFloat));
      p.logsumexp_acc_ptr = logsumexp_acc.data_ptr();
      p.lse_acc_strides = {
          static_cast<int>(logsumexp_acc.stride(0)),
          static_cast<int>(logsumexp_acc.stride(1)),
          static_cast<int>(logsumexp_acc.stride(2)),
          static_cast<int>(logsumexp_acc.stride(3))};
    }
  };

  auto set_grouped_forward_params = [&](GroupedForwardParams& p) {
    p.num_batches = seqstart_q->size(0) - 1;
    p.M = M;
    p.N = N;
    p.Hq = Hq;
    p.Hkv = Hkv;
    p.K = K;
    p.Kv = Kv;

    p.max_seqlen_q = *max_seqlen_q_;

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(K)));
    }

    p.q_ptr = query.data_ptr();
    p.k_ptr = key.data_ptr();
    p.v_ptr = value.data_ptr();
    p.out_ptr = out.data_ptr();

    p.q_strides = {
        static_cast<int>(query.stride(1)),
        static_cast<int>(query.stride(2)),
        static_cast<int>(query.stride(3))};
    p.k_strides = {
        static_cast<int>(key.stride(1)),
        static_cast<int>(key.stride(2)),
        static_cast<int>(key.stride(3))};
    p.v_strides = {
        static_cast<int>(value.stride(1)),
        static_cast<int>(value.stride(2)),
        static_cast<int>(value.stride(3))};
    p.out_strides = {
        static_cast<int>(out.stride(1)),
        static_cast<int>(out.stride(2)),
        static_cast<int>(out.stride(3))};

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      p.attn_bias_ptr = bias->data_ptr();

      const at::Tensor bias_4d_view = get_bias_4d_view(*bias, B, Hq, M, N);
      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};
    } else
      p.has_attn_bias = false;

    p.custom_mask_type = custom_mask_type;
    p.window_size =
        window_size.has_value() ? (*window_size > 0 ? *window_size : 0) : 0;

    // interesting: the tensors have to be defined here, moving to more local
    // scope will cause issue
    at::Tensor dev_seqstart_q;
    at::Tensor dev_seqstart_k;
    at::Tensor dev_seqlen_k;

    p.seqstart_q_dev_ptr = seqstart_q->data_ptr();
    p.seqstart_k_dev_ptr = seqstart_k->data_ptr();

    if (seqlen_k.has_value()) {
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      TORCH_CHECK(seqlen_k->dim() == 1);
      TORCH_CHECK(seqlen_k->size(0) == p.num_batches)
      CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqlen_k));

      p.seqlen_k_dev_ptr = seqlen_k->data_ptr();
    } else
      p.seqlen_k_dev_ptr = nullptr;

    p.is_gappy = false;
    if (block_tables.has_value()) {
      p.block_table_ptr = block_tables->data_ptr();
      p.page_block_size = *page_size;
      p.batch_stride_block_table = block_tables->stride(0);
      p.use_paged_kvcache = true;

      TORCH_CHECK(seqlen_k.has_value());

      // PageBlockDiagonalGappyKeysMask has special way to use seqstart_k,
      // somehow ck_tile kernel need know this
      if (seqstart_k->size(0) == seqlen_k->size(0))
        p.is_gappy = true;
    } else
      p.use_paged_kvcache = false;

    p.philox_seed = philox_seed;
    p.philox_offset = philox_offset;
    p.compute_logsumexp = compute_logsumexp;

    // the following parameters are only used by training forward
    if (use_dropout) {
      p.dropout_prob = static_cast<float>(dropout_p);
    } else
      p.dropout_prob = 0.0f;

    if (p.compute_logsumexp) {
      logsumexp = at::empty({1, Hq, M}, opts.dtype(at::kFloat));
      p.logsumexp_ptr = logsumexp->data_ptr();
      p.lse_strides = {
          static_cast<int>(logsumexp->stride(1)),
          static_cast<int>(logsumexp->stride(2))};
    } else {
      p.logsumexp_ptr = nullptr;
      p.lse_strides = {0, 0};
    }

    bool use_split_kv;
    int num_kv_splits;

    // added for support split_kv
    std::tie(use_split_kv, num_kv_splits) = get_num_kv_splits_heuristic(
        p.num_batches, p.Hq, p.max_seqlen_q, std::max(p.K, p.Kv), 8);

    // 1) fmha fwd split-kv kernel does not support dropout
    // 2) Paged-KVcache is only available from the split-kv kernel at present
    p.use_split_kv =
        (p.use_paged_kvcache || (!use_dropout && use_split_kv)) ? true : false;

    p.num_kv_splits = num_kv_splits;

    if (p.use_split_kv && p.num_kv_splits > 1) {
      out_acc = at::empty({p.num_kv_splits, M, Hq, Kv}, opts.dtype(at::kFloat));
      p.out_acc_ptr = out_acc.data_ptr();
      p.out_acc_strides = {
          static_cast<int>(out_acc.stride(0)),
          static_cast<int>(out_acc.stride(1)),
          static_cast<int>(out_acc.stride(2)),
          static_cast<int>(out_acc.stride(3))};

      logsumexp_acc =
          at::empty({p.num_kv_splits, 1, Hq, M}, opts.dtype(at::kFloat));
      p.logsumexp_acc_ptr = logsumexp_acc.data_ptr();
      p.lse_acc_strides = {
          static_cast<int>(logsumexp_acc.stride(0)),
          static_cast<int>(logsumexp_acc.stride(2)),
          static_cast<int>(logsumexp_acc.stride(3))};
    }
  };

  auto inDataType = query.scalar_type();

  if (!seqstart_q.has_value()) { // input is batched
    BatchedForwardParams batched_forward_params;

    set_batched_forward_params(batched_forward_params);

    if (!batched_forward_params.compute_logsumexp) {
      if (inDataType == at::ScalarType::Half) {
        batched_infer_fp16(batched_forward_params, stream);
      } else if (inDataType == at::ScalarType::BFloat16) {
        batched_infer_bf16(batched_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    } else {
      if (inDataType == at::ScalarType::Half) {
        batched_forward_fp16(batched_forward_params, stream);
      } else if (inDataType == at::ScalarType::BFloat16) {
        batched_forward_bf16(batched_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    };
  } else { // input is grouped
    GroupedForwardParams grouped_forward_params;

    set_grouped_forward_params(grouped_forward_params);

    if (!grouped_forward_params.compute_logsumexp) {
      if (inDataType == at::ScalarType::Half) {
        grouped_infer_fp16(grouped_forward_params, stream);
      } else if (inDataType == at::ScalarType::BFloat16) {
        grouped_infer_bf16(grouped_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    } else {
      if (inDataType == at::ScalarType::Half) {
        grouped_forward_fp16(grouped_forward_params, stream);
      } else if (inDataType == at::ScalarType::BFloat16) {
        grouped_forward_bf16(grouped_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    };
  };

  return std::make_tuple(out, logsumexp, philox_seed, philox_offset);
}

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<at::Tensor, std::optional<at::Tensor>, int64_t, int64_t>
efficient_attention_forward_ck_meta(
    const at::Tensor& query, // [b, seqlen, num_heads_q, K]
    const at::Tensor& key, // [b, seqlen, num_heads_kv, K]
    const at::Tensor& value, // [b, seqlen, num_heads_kv, Kv]
    const c10::optional<at::Tensor>& bias, // [b, num_heads_q, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    double dropout_p, // attention matrix dropout probability
    bool compute_logsumexp,
    int64_t custom_mask_type,
    c10::optional<double> scale,
    const c10::optional<at::Tensor>& seqlen_k,
    const c10::optional<int64_t> window_size,
    const c10::optional<at::Tensor>& block_tables,
    const c10::optional<int64_t> page_size) {
  at::SymInt B = query.sym_size(0);
  at::SymInt M = query.sym_size(1);
  at::SymInt N = key.sym_size(1);
  at::SymInt Hq = query.sym_size(-2);
  at::SymInt Hkv = key.sym_size(-2);
  at::SymInt K = query.sym_size(-1);
  at::SymInt Kv = value.sym_size(-1);
  auto opts = query.options();
  std::optional<at::Tensor> logsumexp = std::nullopt;
  at::Tensor out = at::empty_symint({B, M, Hq, Kv}, opts);
  int64_t philox_seed = 0;
  int64_t philox_offset = 0;
  if (!seqstart_q.has_value()) { // input is batched
    if (compute_logsumexp) {
      logsumexp = at::empty_symint({B, Hq, M}, opts.dtype(at::kFloat));
    }
  } else {
    if (compute_logsumexp) {
      logsumexp = at::empty_symint({1, Hq, M}, opts.dtype(at::kFloat));
    }
  }
  return std::make_tuple(out, logsumexp, philox_seed, philox_offset);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_ck"),
      TORCH_FN(efficient_attention_forward_ck));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_ck"),
      TORCH_FN(efficient_attention_forward_ck_meta));
}
