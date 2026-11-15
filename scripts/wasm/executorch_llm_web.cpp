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

val result_error(const std::string& message) {
  val out = val::object();
  out.set("error", message);
  return out;
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
  // and frees it on delete(). Returns "" on success, else the reason.
  std::string load(
      uintptr_t model_ptr,
      size_t model_size,
      const std::string& tokenizer_path) {
    if (module_) {
      return "A model is already loaded";
    }
    model_data_ = reinterpret_cast<void*>(model_ptr);

    tokenizer_ = el::load_tokenizer(tokenizer_path);
    if (!tokenizer_ || !tokenizer_->is_loaded()) {
      return "Could not load the tokenizer";
    }

    module_ = std::make_unique<Module>(
        std::make_unique<BufferDataLoader>(model_data_, model_size));

    auto metadata = el::get_llm_metadata(tokenizer_.get(), module_.get());
    if (!metadata.ok()) {
      return error_text("Could not read the model's LLM metadata", metadata.error());
    }
    metadata_ = metadata.get();
    if (!metadata_.at(el::kUseKVCache)) {
      return "Only KV-cache exports are supported on web";
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
      return error_text("Could not load the model's forward method", err);
    }
    err = io_manager_->load();
    if (err != Error::Ok) {
      return error_text("Could not set up the model's inputs", err);
    }
    return "";
  }

  // Encodes and prefills [prompt]. Returns {pieces: string[], done: bool} with
  // the echoed prompt (if asked) and the first generated piece, or {error}.
  val begin(
      const std::string& prompt,
      int max_new_tokens,
      int seq_len,
      float temperature,
      bool echo,
      bool ignore_eos,
      int num_bos,
      int num_eos) {
    if (!module_) {
      return result_error("No model is loaded");
    }
    done_ = true;
    if (prompt.empty()) {
      return result_error("The prompt is empty");
    }
    if (temperature < 0.0f || temperature > 1.0f) {
      return result_error("Temperature must be in [0, 1]");
    }

    auto encoded = tokenizer_->encode(prompt, num_bos, num_eos);
    if (!encoded.ok()) {
      return result_error("Could not tokenize the prompt");
    }
    std::vector<uint64_t> tokens = encoded.get();
    const auto num_prompt_tokens = static_cast<int64_t>(tokens.size());
    if (num_prompt_tokens < 1) {
      return result_error("The prompt has no tokens");
    }

    const int64_t max_seq_len = metadata_.at(el::kMaxSeqLen);
    const int64_t max_context_len = metadata_.at(el::kMaxContextLen);
    if (max_seq_len >= max_context_len) {
      if (pos_ + num_prompt_tokens >= max_context_len) {
        return result_error(
            "The conversation no longer fits the model's context; reset it");
      }
    } else if (num_prompt_tokens >= max_context_len) {
      return result_error("The prompt is longer than the model's context");
    }

    val pieces = val::array();
    if (echo) {
      pieces.call<void>("push", prompt);
    }

    auto prefilled = prefiller_->prefill(tokens, pos_);
    if (!prefilled.ok()) {
      return result_error(error_text("Prefill failed", prefilled.error()));
    }
    cur_token_ = prefilled.get();

    el::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.seq_len = seq_len;
    const int64_t occupied = max_seq_len < max_context_len ? 0 : pos_;
    remaining_ = config.resolve_max_new_tokens(max_context_len, occupied);
    if (remaining_ <= 0) {
      return result_error("No room left in the model's context; reset it");
    }

    auto first = tokenizer_->decode(cur_token_, cur_token_);
    if (!first.ok()) {
      return result_error("Could not decode the first token");
    }
    pieces.call<void>("push", first.get());

    temperature_ = temperature;
    ignore_eos_ = ignore_eos;
    token_data_.assign(1, static_cast<int64_t>(cur_token_));
    tokens_ = executorch::extension::from_blob(
        token_data_.data(), {1, 1}, ScalarType::Long);
    // Prefill produced one token; the rest come from step().
    remaining_ -= 1;
    done_ = remaining_ <= 0;

    val out = val::object();
    out.set("pieces", pieces);
    out.set("done", done_);
    return out;
  }

  // Generates one token. Returns {piece: string, done: bool} or {error}.
  val step() {
    val out = val::object();
    if (done_ || !module_) {
      out.set("piece", std::string());
      out.set("done", true);
      return out;
    }
    auto logits = decoder_->step(tokens_, pos_);
    if (!logits.ok()) {
      done_ = true;
      return result_error(error_text("Decoding failed", logits.error()));
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
      return result_error("Could not decode a token");
    }
    done_ = remaining_ <= 0 ||
        (!ignore_eos_ && eos_ids_.count(cur_token_) > 0);
    out.set("piece", piece.get());
    out.set("done", done_);
    return out;
  }

  // Starts a new conversation: the next prompt is written from position 0.
  void reset() {
    pos_ = 0;
    done_ = true;
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
  uint64_t cur_token_ = 0;
  int64_t remaining_ = 0;
  float temperature_ = 0.0f;
  bool ignore_eos_ = false;
  bool done_ = true;
  std::vector<int64_t> token_data_;
  TensorPtr tokens_;
};

} // namespace

EMSCRIPTEN_BINDINGS(executorch_llm_web) {
  emscripten::class_<WebLLMRunner>("LLMRunner")
      .constructor<>()
      .function("load", &WebLLMRunner::load)
      .function("begin", &WebLLMRunner::begin)
      .function("step", &WebLLMRunner::step)
      .function("reset", &WebLLMRunner::reset);
}
