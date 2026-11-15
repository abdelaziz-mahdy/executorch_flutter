// ExecuTorch LLM runner for the web (Embind).
//
// Mirrors the native runner the Dart FFI layer builds (text decoder + prefiller
// + tokenizer + the model's stop set), but hands out one token per call instead
// of looping inside generate(). A browser worker has no shared-memory stop flag
// without cross-origin isolation, so the worker drives the loop itself and
// yields to its message queue between tokens; that is what lets stop() and new
// requests through while a reply is streaming.
//
// Built into executorch_llm.{js,wasm} by scripts/build_wasm_in_container.sh and
// used only from web/js/executorch_llm_worker.js.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

namespace {

using emscripten::val;
using executorch::aten::ScalarType;
using executorch::extension::BufferDataLoader;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
namespace el = executorch::extension::llm;

std::string error_text(const char* what, Error err) {
  char code[32];
  snprintf(code, sizeof(code), " (ExecuTorch error 0x%x)", static_cast<int>(err));
  return std::string(what) + code;
}

// The exception class Dart throws, mirroring the native FFI's error mapping
// (to_et_error + _mapErrorCode): argument errors are validation errors, state
// errors are model errors, the rest are inference errors.
const char* error_kind(Error err) {
  switch (err) {
    case Error::InvalidArgument:
      return "validation";
    case Error::MemoryAllocationFailed:
      return "memory";
    case Error::InvalidState:
      return "state";
    case Error::NotSupported:
      return "unsupported";
    default:
      return "inference";
  }
}

val result_error(const std::string& message, const char* kind) {
  val out = val::object();
  out.set("error", message);
  out.set("kind", std::string(kind));
  return out;
}

val result_error(const std::string& message, Error err) {
  return result_error(message, error_kind(err));
}

// Runs [body], turning a C++ exception into an error result. Exceptions are
// only a backstop (ExecuTorch reports errors as values), but one escaping
// into JS would arrive as a bare pointer. The native FFI catches them at its
// boundary the same way.
template <typename Body>
val guarded(Body&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    return result_error(std::string("Unexpected error: ") + e.what(), "inference");
  } catch (...) {
    return result_error("Unexpected error", "inference");
  }
}

// A JS copy of [bytes]. Token pieces go to JS as bytes, not strings: a piece
// can end partway through a UTF-8 character, and the worker's streaming
// TextDecoder joins and validates them across pieces.
val js_bytes(const std::string& bytes) {
  return val::global("Uint8Array")
      .new_(emscripten::typed_memory_view(
          bytes.size(), reinterpret_cast<const uint8_t*>(bytes.data())));
}

class WebLLMRunner {
 public:
  WebLLMRunner() = default;
  WebLLMRunner(const WebLLMRunner&) = delete;
  WebLLMRunner& operator=(const WebLLMRunner&) = delete;

  ~WebLLMRunner() {
    // The module reads the model bytes until it is destroyed, so free them
    // after it.
    prefiller_.reset();
    decoder_.reset();
    io_manager_.reset();
    module_.reset();
    std::free(model_data_);
  }

  // Loads the model from [model_ptr, model_ptr + model_size), a buffer the
  // worker allocated with _malloc and filled; the runner owns it from here on
  // and frees it on delete(). Returns {} on success, else {error, kind}.
  val load_impl(
      uintptr_t model_ptr,
      size_t model_size,
      const std::string& tokenizer_path) {
    if (module_) {
      return result_error("A model is already loaded", Error::InvalidState);
    }
    model_data_ = reinterpret_cast<void*>(model_ptr);

    tokenizer_ = el::load_tokenizer(tokenizer_path);
    if (!tokenizer_ || !tokenizer_->is_loaded()) {
      return result_error("Could not load the tokenizer", Error::InvalidArgument);
    }

    module_ = std::make_unique<Module>(
        std::make_unique<BufferDataLoader>(model_data_, model_size));

    auto metadata = el::get_llm_metadata(tokenizer_.get(), module_.get());
    if (!metadata.ok()) {
      return result_error(
          error_text("Could not read the model's LLM metadata", metadata.error()),
          metadata.error());
    }
    metadata_ = metadata.get();
    if (!metadata_.at(el::kUseKVCache)) {
      return result_error(
          "Only KV-cache exports are supported on web", Error::NotSupported);
    }

    eos_ids_ = el::get_eos_ids(tokenizer_.get(), module_.get());
    // optimum names the method get_eos_id (singular); the native runner merges
    // it too, so the declared end-of-turn token stops generation here as well.
    auto names = module_->method_names();
    if (names.ok() && names.get().count("get_eos_id")) {
      auto r = module_->execute("get_eos_id");
      if (r.ok()) {
        for (const auto& v : r.get()) {
          if (v.isScalar()) {
            eos_ids_.insert(static_cast<uint64_t>(v.toScalar().to<int64_t>()));
          }
        }
      }
    }

    // Same prefill choice as the native runner: parallel when forward accepts
    // more than one token, one token at a time otherwise.
    bool parallel_prefill = false;
    auto meta = module_->method_meta(el::kForwardMethod);
    if (meta.ok() && meta->num_inputs() > 0) {
      auto in0 = meta->input_tensor_meta(0);
      if (in0.ok() && in0->sizes().size() >= 2 && in0->sizes()[1] > 1) {
        parallel_prefill = true;
      }
    }

    io_manager_ = std::make_unique<el::IOManager>(*module_);
    const auto vocab_size = static_cast<int32_t>(metadata_.at(el::kVocabSize));
    decoder_ = std::make_unique<el::TextDecoderRunner>(
        module_.get(),
        io_manager_.get(),
        el::kForwardMethod,
        std::make_unique<el::Sampler>(vocab_size, 0.0f));
    prefiller_ = std::make_unique<el::TextPrefiller>(
        decoder_.get(),
        /*use_kv_cache=*/true,
        parallel_prefill,
        metadata_.at(el::kMaxSeqLen));

    Error err = prefiller_->load();
    if (err != Error::Ok) {
      return result_error(
          error_text("Could not load the model's forward method", err), err);
    }
    err = io_manager_->load();
    if (err != Error::Ok) {
      return result_error(
          error_text("Could not set up the model's inputs", err), err);
    }
    return val::object();
  }

  // Checks the settings and encodes [prompt] for prefill(). Returns {} or
  // {error, kind}. Split from prefill() so the worker can echo the prompt
  // where upstream does: after these checks, before prefill.
  val start_impl(
      const std::string& prompt,
      int max_new_tokens,
      int seq_len,
      float temperature,
      bool ignore_eos,
      int num_bos,
      int num_eos) {
    if (!module_) {
      return result_error("No model is loaded", Error::InvalidState);
    }
    done_ = true;
    started_ = false;
    if (prompt.empty()) {
      // Upstream: "Empty prompt requires a prior prefill() call".
      return result_error("The prompt is empty", Error::InvalidState);
    }
    // Written so NaN fails too, as upstream's check does.
    if (!(temperature >= 0.0f && temperature <= 1.0f)) {
      return result_error(
          "Temperature must be in [0, 1]", Error::InvalidArgument);
    }
    if (max_new_tokens == 0 || max_new_tokens < -1) {
      return result_error(
          "maxNewTokens must be -1 (fill the context) or at least 1",
          Error::InvalidArgument);
    }

    auto encoded = tokenizer_->encode(prompt, num_bos, num_eos);
    if (!encoded.ok()) {
      return result_error(
          "Could not tokenize the prompt", Error::InvalidArgument);
    }
    prompt_tokens_ = encoded.get();
    const auto num_prompt_tokens = static_cast<int64_t>(prompt_tokens_.size());
    if (num_prompt_tokens < 1) {
      return result_error("The prompt has no tokens", Error::InvalidArgument);
    }

    const int64_t max_seq_len = metadata_.at(el::kMaxSeqLen);
    const int64_t max_context_len = metadata_.at(el::kMaxContextLen);
    if (max_seq_len >= max_context_len) {
      if (pos_ + num_prompt_tokens >= max_context_len) {
        return result_error(
            "The conversation no longer fits the model's context; reset it",
            Error::InvalidArgument);
      }
    } else if (num_prompt_tokens >= max_context_len) {
      return result_error(
          "The prompt is longer than the model's context",
          Error::InvalidArgument);
    }

    max_new_tokens_ = max_new_tokens;
    seq_len_ = seq_len;
    temperature_ = temperature;
    ignore_eos_ = ignore_eos;
    started_ = true;
    return val::object();
  }

  // Prefills the prompt start() encoded. Returns {bytes: Uint8Array, done:
  // bool} with the first generated piece's bytes, or {error, kind}.
  val prefill_impl() {
    if (!started_) {
      return result_error("Call start() first", Error::InvalidState);
    }
    started_ = false;
    const int64_t max_seq_len = metadata_.at(el::kMaxSeqLen);
    const int64_t max_context_len = metadata_.at(el::kMaxContextLen);

    auto prefilled = prefiller_->prefill(prompt_tokens_, pos_);
    prompt_tokens_.clear();
    if (!prefilled.ok()) {
      return result_error(
          error_text("Prefill failed", prefilled.error()), prefilled.error());
    }
    cur_token_ = prefilled.get();

    el::GenerationConfig config;
    config.max_new_tokens = max_new_tokens_;
    config.seq_len = seq_len_;
    const int64_t occupied = max_seq_len < max_context_len ? 0 : pos_;
    remaining_ = config.resolve_max_new_tokens(max_context_len, occupied);
    if (remaining_ <= 0) {
      return result_error(
          "No room left in the model's context; reset it",
          Error::InvalidArgument);
    }

    auto first = tokenizer_->decode(cur_token_, cur_token_);
    if (!first.ok()) {
      return result_error(
          "Could not decode the first token", Error::InvalidArgument);
    }

    // Upstream adds the generated tokens to pos_ only when generation
    // succeeds (or stops); after an error it stays where prefill left it.
    generation_start_pos_ = pos_;
    token_data_.assign(1, static_cast<int64_t>(cur_token_));
    tokens_ = executorch::extension::from_blob(
        token_data_.data(), {1, 1}, ScalarType::Long);
    // Prefill produced one token; the rest come from step().
    remaining_ -= 1;
    done_ = remaining_ <= 0;

    val out = val::object();
    out.set("bytes", js_bytes(first.get()));
    out.set("done", done_);
    return out;
  }

  // Generates one token. Returns {bytes: Uint8Array, done: bool} or
  // {error, kind}.
  val step_impl() {
    val out = val::object();
    if (done_ || !module_) {
      out.set("bytes", js_bytes(std::string()));
      out.set("done", true);
      return out;
    }
    auto logits = decoder_->step(tokens_, pos_);
    if (!logits.ok()) {
      done_ = true;
      pos_ = generation_start_pos_;
      return result_error(
          error_text("Decoding failed", logits.error()), logits.error());
    }
    const uint64_t prev = cur_token_;
    cur_token_ = static_cast<uint64_t>(
        decoder_->logits_to_token(logits.get(), temperature_));
    pos_ += 1;
    token_data_[0] = static_cast<int64_t>(cur_token_);
    remaining_ -= 1;

    auto piece = tokenizer_->decode(prev, cur_token_);
    if (!piece.ok()) {
      done_ = true;
      pos_ = generation_start_pos_;
      return result_error("Could not decode a token", Error::InvalidArgument);
    }
    done_ = remaining_ <= 0 ||
        (!ignore_eos_ && eos_ids_.count(cur_token_) > 0);
    out.set("bytes", js_bytes(piece.get()));
    out.set("done", done_);
    return out;
  }

  // Starts a new conversation: the next prompt is written from position 0.
  void reset() {
    pos_ = 0;
    done_ = true;
    started_ = false;
  }

 public:
  val load(uintptr_t model_ptr, size_t model_size, const std::string& tokenizer_path) {
    return guarded([&] { return load_impl(model_ptr, model_size, tokenizer_path); });
  }

  val start(
      const std::string& prompt,
      int max_new_tokens,
      int seq_len,
      float temperature,
      bool ignore_eos,
      int num_bos,
      int num_eos) {
    return guarded([&] {
      return start_impl(
          prompt, max_new_tokens, seq_len, temperature, ignore_eos, num_bos, num_eos);
    });
  }

  val prefill() {
    return guarded([&] { return prefill_impl(); });
  }

  val step() {
    val out = guarded([&] { return step_impl(); });
    if (out.hasOwnProperty("error")) {
      // Whatever failed, end the generation as step_impl's own errors do.
      done_ = true;
      pos_ = generation_start_pos_;
    }
    return out;
  }

 private:
  void* model_data_ = nullptr;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<el::IOManager> io_manager_;
  std::unique_ptr<el::TextDecoderRunner> decoder_;
  std::unique_ptr<el::TextPrefiller> prefiller_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  int64_t pos_ = 0;
  int64_t generation_start_pos_ = 0;
  uint64_t cur_token_ = 0;
  int64_t remaining_ = 0;
  float temperature_ = 0.0f;
  bool ignore_eos_ = false;
  bool done_ = true;
  bool started_ = false;
  std::vector<uint64_t> prompt_tokens_;
  int max_new_tokens_ = -1;
  int seq_len_ = -1;
  std::vector<int64_t> token_data_;
  TensorPtr tokens_;
};

} // namespace

EMSCRIPTEN_BINDINGS(executorch_llm_web) {
  emscripten::class_<WebLLMRunner>("LLMRunner")
      .constructor<>()
      .function("load", &WebLLMRunner::load)
      .function("start", &WebLLMRunner::start)
      .function("prefill", &WebLLMRunner::prefill)
      .function("step", &WebLLMRunner::step)
      .function("reset", &WebLLMRunner::reset);
}
