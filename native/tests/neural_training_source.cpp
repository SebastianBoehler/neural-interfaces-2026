#include "../src/training_internal.h"
#include "ni26/math.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
using namespace ni26;
namespace {
double maximum = 0;
void same(const std::string &name, const Tensor &x, const Tensor &y, double tolerance) {
  require(x.sizes() == y.sizes(), "source training shape differs: " + name);
  finite(x, name);
  auto error =
      (x.detach().cpu().to(torch::kFloat64) - y.to(torch::kFloat64)).abs().max().item<double>();
  maximum = std::max(maximum, error);
  std::cout << name << " maxabs=" << std::setprecision(17) << error << '\n';
  require(error <= tolerance, "source training oracle mismatch: " + name);
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: neural_training_source SOURCE_FIXTURE");
    torch::set_num_threads(1);
    auto fixture = Params::load(argv[1]);
    require(fixture.integer("source_only") == 1, "source-only fixture required");
    neural::RawFBC net(fixture.subset("initial_"), torch::kCPU);
    auto parameters = net.parameters();
    auto body_before = parameters[0].detach().clone(), head_before = parameters[4].detach().clone();
    torch::optim::AdamW opt(parameters, torch::optim::AdamWOptions(.001).weight_decay(.001));
    auto schedule = fixture.get("expected_schedule");
    require(schedule.sizes() == torch::IntArrayRef({1240, 2}), "full scheduler fixture required");
    std::vector<Tensor> actual_schedule;
    for (int64_t i = 0; i < 1240; ++i) {
      auto [lr, beta] = neural::one_cycle(i, 1240);
      actual_schedule.push_back(torch::tensor({lr, beta}, torch::kFloat64));
    }
    same("OneCycle", torch::stack(actual_schedule), schedule, 1e-15);
    auto [lr, beta] = neural::one_cycle(0, 1240);
    auto &options = static_cast<torch::optim::AdamWOptions &>(opt.param_groups()[0].options());
    options.lr(lr);
    options.betas(std::make_tuple(beta, .999));
    auto count = fixture.get("batch_sizes")[0].item<int64_t>();
    auto ids = fixture.get("batch_indices")[0].slice(0, 0, count);
    auto x = fixture.get("source_X").index_select(0, ids),
         y = fixture.get("source_y").index_select(0, ids);
    opt.zero_grad();
    auto begun = std::chrono::steady_clock::now();
    auto logits = net.forward(x, true);
    auto forward_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
    same("first_logits", logits, fixture.get("first_logits"), 1e-5);
    same("first_posterior", torch::softmax(logits.to(torch::kFloat64), 1),
         torch::softmax(fixture.get("first_logits").to(torch::kFloat64), 1), 1e-6);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    same("first_loss", loss, fixture.get("first_loss"), 1e-6);
    begun = std::chrono::steady_clock::now();
    loss.backward();
    auto backward_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
    const std::vector<std::string> names{"spatial_conv.0.parametrizations.weight.original",
                                         "spatial_conv.0.bias",
                                         "spatial_conv.1.weight",
                                         "spatial_conv.1.bias",
                                         "final_layer.parametrizations.weight.original",
                                         "final_layer.bias"};
    for (size_t i = 0; i < parameters.size(); ++i) {
      require(parameters[i].grad().defined(), "native source gradient inactive");
      same("gradient/" + names[i], parameters[i].grad(), fixture.get("first_gradient_" + names[i]),
           1e-5);
    }
    begun = std::chrono::steady_clock::now();
    opt.step();
    require((parameters[0].detach() - body_before).abs().max().item<double>() > 0 &&
                (parameters[4].detach() - head_before).abs().max().item<double>() > 0,
            "native first source body/head optimizer update inactive");
    auto optimizer_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begun).count();
    for (const auto &[name, value] : net.state().tensors)
      same("state/" + name, value, fixture.get("first_state_" + name), 1e-6);
    std::cout << "NEURAL_FIRST_SOURCE_UPDATE_PASS maxabs=" << maximum << '\n';
    std::cout << "native_forward_seconds=" << forward_seconds
              << " native_backward_seconds=" << backward_seconds
              << " native_optimizer_seconds=" << optimizer_seconds << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
