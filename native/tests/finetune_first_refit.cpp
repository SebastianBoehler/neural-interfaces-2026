#include "../src/finetune_adam.h"
#include "../src/networks_internal.h"
#include <iostream>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/torch.h>
using namespace ni26;
namespace {
Tensor texts(const std::vector<std::string> &names) {
  size_t width = 1;
  for (const auto &name : names)
    width = std::max(width, name.size() + 1);
  auto result =
      torch::zeros({int64_t(names.size()), int64_t(width)}, torch::kUInt8);
  for (size_t i = 0; i < names.size(); ++i)
    std::copy(names[i].begin(), names[i].end(), result[i].data_ptr<uint8_t>());
  return result;
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 5,
            "usage: finetune_first_refit PARAMS FIXTURE MODULE_ROOT OUTPUT");
    torch::set_num_threads(1);
    auto z = Params::load(argv[1]), f = Params::load(argv[2]);
    auto slot = z.texts("v6_slots");
    require(slot.front() == "fbraw", "expected original fbraw firstslot");
    int64_t member = z.get("v6_ft_member")[0].item<int64_t>(), column = -1;
    auto ids = f.get("input_ids");
    for (int64_t i = 0; i < ids.numel(); ++i)
      if (ids[i].item<int64_t>() == member)
        column = i;
    require(column >= 0, "RAWsource input missing");
    auto x =
        f.get("unknown_input").slice(0, 0, 20).select(1, column).contiguous();
    auto current = f.get("unknown_input")[20][column].unsqueeze(0);
    auto labels = f.get("unknown_relabel_20").argmax(1),
         weights = std::get<0>(f.get("unknown_relabel_20").max(1))
                       .to(torch::kFloat32);
    auto model =
        torch::jit::load((std::filesystem::path(argv[3]) /
                          z.text("net" + std::to_string(member) + "_sjit_file"))
                             .string(),
                         torch::kCPU);
    auto parameters = neural::trainable(model);
    for (auto &p : parameters)
      p.requires_grad_(true);
    std::vector<std::string> names, buffers;
    for (auto item : model.named_parameters(true))
      if (item.value.requires_grad())
        names.push_back(item.name);
    for (auto item : model.named_buffers(true)) {
      auto pos = item.name.find_last_of('.');
      auto leaf = item.name.substr(pos == std::string::npos ? 0 : pos + 1);
      if (leaf == "w" || leaf == "b")
        names.push_back(item.name);
      buffers.push_back(item.name);
    }
    require(names.size() == parameters.size(),
            "actual RAWparameter order differs");
    Params out;
    out.tensors["native_format"] = torch::tensor(1, torch::kInt64);
    out.tensors["parameter_names"] = texts(names);
    out.tensors["buffer_names"] = texts(buffers);
    auto save = [&](const std::string &key, const Tensor &t) {
      out.tensors[key] = t.detach().cpu().clone();
    };
    auto save_buffers = [&](const std::string &prefix) {
      int i = 0;
      for (auto item : model.named_buffers(true))
        save(prefix + std::to_string(i++), item.value);
    };
    save("X", x);
    save("labels", labels);
    save("weights", weights);
    auto gen = neural::generator(torch::kCPU, z.integer("v6_ft_seed"));
    save("generator_before", gen.get_state());
    {
      torch::AutoGradMode enable(true);
      torch::jit::GraphOptimizerEnabledGuard optimizer_guard(false);
      neural::FineTuneAdam optimizer(parameters, z.number("v6_ft_lr"));
      for (int step = 0; step < 5; ++step) {
        auto prefix = "step" + std::to_string(step) + "_";
        optimizer.zero_grad();
        neural::zero_stream(model);
        for (size_t i = 0; i < parameters.size(); ++i)
          save(prefix + "before_" + std::to_string(i), parameters[i]);
        Tensor clean;
        {
          torch::NoGradGuard guard;
          clean = model.forward({x}).toTensor().slice(1, 0, 3);
        }
        save(prefix + "clean", clean);
        save_buffers(prefix + "clean_buffer_");
        neural::zero_stream(model);
        auto augmented = neural::strong(x, gen);
        save(prefix + "augmented", augmented);
        save(prefix + "generator_after", gen.get_state());
        auto logits = model.forward({augmented}).toTensor().slice(1, 0, 3);
        save(prefix + "logits", logits);
        save_buffers(prefix + "aug_buffer_");
        namespace F = torch::nn::functional;
        auto ce =
            (weights * F::cross_entropy(logits, labels,
                                        F::CrossEntropyFuncOptions().reduction(
                                            torch::kNone)))
                .sum() /
            weights.sum().clamp_min(1e-6);
        auto kl =
            F::kl_div(torch::log_softmax(logits, 1), torch::softmax(clean, 1),
                      F::KLDivFuncOptions().reduction(torch::kBatchMean));
        auto loss = ce + z.number("v6_ft_lam") * kl;
        save(prefix + "ce", ce);
        save(prefix + "kl", kl);
        save(prefix + "loss", loss);
        loss.backward();
        for (size_t i = 0; i < parameters.size(); ++i) {
          require(parameters[i].grad().defined(), "RAWgradient absent");
          save(prefix + "gradient_" + std::to_string(i), parameters[i].grad());
        }
        optimizer.step();
        for (size_t i = 0; i < parameters.size(); ++i) {
          save(prefix + "after_" + std::to_string(i), parameters[i]);
          save(prefix + "m_" + std::to_string(i), optimizer.first(i));
          save(prefix + "v_" + std::to_string(i), optimizer.second(i));
          save(prefix + "adam_step_" + std::to_string(i),
               torch::tensor(optimizer.steps(i), torch::kInt64));
        }
      }
      {
        torch::NoGradGuard guard;
        neural::zero_stream(model);
        model.forward({x});
        save_buffers("rebuilt_buffer_");
      }
    }
    {
      torch::NoGradGuard guard;
      save("current_logits",
           model.forward({current}).toTensor().slice(1, 0, 3));
      save("current_p",
           torch::softmax(out.get("current_logits").to(torch::kFloat64), 1)[0]);
      save_buffers("current_buffer_");
    }
    out.save(argv[4]);
    std::cout << "RAW_FIRST_REFIT_TRACE_COMPLETE steps=5\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
