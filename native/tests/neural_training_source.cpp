#include "../src/training_internal.h"
#include "ni26/initialization.h"
#include "ni26/math.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
using namespace ni26;
namespace {
double maximum = 0;
bool numeric_failed = false;
void same(const std::string &name, const Tensor &x, const Tensor &y,
          double tolerance) {
  require(x.sizes() == y.sizes(), "source training shape differs: " + name);
  finite(x, name);
  auto error = (x.detach().cpu().to(torch::kFloat64) - y.to(torch::kFloat64))
                   .abs()
                   .max()
                   .item<double>();
  maximum = std::max(maximum, error);
  std::cout << name << " maxabs=" << std::setprecision(17) << error << '\n';
  if (error > tolerance) {
    std::cerr << "BOUND_EXCESS " << name << " maxabs=" << error
              << " bound=" << tolerance << '\n';
    numeric_failed = true;
  }
}
void compare_moment(const std::string &name, const Tensor &actual,
                    const Tensor &expected) {
  same(name, actual, expected, 1e-7);
  auto reference = expected.to(torch::kFloat64);
  double norm = reference.norm().item<double>();
  double error = (actual.to(torch::kFloat64) - reference).norm().item<double>();
  double relative = norm > 0 ? error / norm : error;
  std::cout << name << " relative_l2=" << relative << '\n';
  if (relative > 1e-6) {
    std::cerr << "BOUND_EXCESS " << name << " relative_l2=" << relative << '\n';
    numeric_failed = true;
  }
}

void compare_prefix_step(const Params &oracle, neural::RawFBC &net,
                         torch::optim::AdamW &optimizer,
                         const neural::SourceUpdate &update,
                         const std::vector<std::string> &names, int64_t step) {
  auto prefix = std::to_string(step) + "_";
  auto field = [&](const std::string &key) { return oracle.get(prefix + key); };
  auto &options = static_cast<torch::optim::AdamWOptions &>(
      optimizer.param_groups()[0].options());
  same(prefix + "schedule",
       torch::tensor({options.lr(), std::get<0>(options.betas())}, torch::kFloat64),
       field("schedule"), 1e-15);
  same(prefix + "logits", update.logits, field("logits"), 1e-5);
  same(prefix + "posterior", update.logits.to(torch::kFloat64).softmax(1),
       field("logits").to(torch::kFloat64).softmax(1), 1e-6);
  same(prefix + "loss", update.loss, field("loss"), 1e-6);
  for (const auto &[name, value] : net.state().tensors)
    same(prefix + "state/" + name, value, field("state_" + name), 1e-6);
  auto parameters = net.parameters();
  require(optimizer.state().size() == names.size(), "AdamW state count differs");
  for (size_t i = 0; i < names.size(); ++i) {
    const auto &name = names[i];
    auto &state = optimizer.state().at(parameters[i].unsafeGetTensorImpl());
    auto *adam = dynamic_cast<torch::optim::AdamWParamState *>(state.get());
    require(adam && adam->step() == step &&
                adam->step() == field("step_" + name).item<int64_t>(),
            "AdamW state step differs: " + name);
    same(prefix + "gradient/" + name, parameters[i].grad(),
         field("gradient_" + name), 1e-5);
    compare_moment(prefix + "exp_avg/" + name, adam->exp_avg(),
                   field("exp_avg_" + name));
    compare_moment(prefix + "exp_avg_sq/" + name, adam->exp_avg_sq(),
                   field("exp_avg_sq_" + name));
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3 || argc == 4, "usage: neural_training_source "
                       "SOURCE_PREPARATION_FIXTURE UPDATE_ORACLE [PREFIX_ORACLE]");
    torch::set_num_threads(1);
    auto raw = Params::load(argv[1]), fixture = Params::load(argv[2]);
    require(!raw.has("y"),
            "original source preparation fixture must remain label-free");
    Params input;
    for (const auto &key : {"x", "stream", "person", "fit_mask", "held_person"})
      input.tensors[key] = raw.get(key);
    for (const auto &key : {"source_only", "epochs", "initialization_seed",
                            "source_people", "excluded_people"})
      input.tensors[key] = fixture.get(key);
    input.tensors["y"] = torch::full({raw.get("x").size(0)}, -1, torch::kInt64);
    input.tensors["y"].index_copy_(0, fixture.get("source_rows"),
                                   fixture.get("source_y"));
    auto original = fixture.subset("initial_");
    for (int i = 0; i < 9; ++i) {
      auto key = "spectral_filtering.fir_list." + std::to_string(i);
      input.tensors["fir_" + key] = original.get(key);
    }
    auto prepared = neural::prepare_raw_training_source(input);
    require(torch::equal(prepared.source.rows, fixture.get("source_rows")),
            "native source order differs");
    require(
        torch::equal(prepared.source.session, fixture.get("source_session")),
        "native source session differs");
    require(torch::equal(prepared.labels, fixture.get("source_y")),
            "native source labels differ");
    require(torch::equal(prepared.source.x, fixture.get("source_X")),
            "native source X not BYTE_EXACT");
    same("source_G", prepared.source.covariance_prior, raw.get("expected_G"),
         1e-8);
    require(std::abs(prepared.source.scale - raw.number("expected_scale")) <=
                1e-10,
            "native source scale differs");
    auto initialized = prepared.initial;
    require(initialized.tensors.size() == original.tensors.size() + 1,
            "initializer state fields differ");
    for (const auto &[key, value] : original.tensors)
      require(torch::equal(initialized.get(key), value),
              "source initializer differs: " + key);
    std::cout
        << "SOURCE_NATIVE_PREPARATION_INITIALIZER_BYTE_EXACT rows=1898 fields="
        << original.tensors.size() << '\n';
    int64_t index = 0;
    for (const auto &epoch : prepared.epochs)
      for (const auto &batch : epoch) {
        auto count = fixture.get("batch_sizes")[index].item<int64_t>();
        require(int64_t(batch.size()) == count, "native batch size differs");
        auto ids = torch::tensor(batch, torch::kInt64);
        require(torch::equal(ids, fixture.get("batch_indices")[index].slice(
                                      0, 0, count)),
                "native batch rows differ");
        auto sessions =
            prepared.source.session.index_select(0, ids).reshape({-1, 32});
        require(
            (sessions == sessions.select(1, 0).unsqueeze(1)).all().item<bool>(),
            "native chunk crosses sessions");
        ++index;
      }
    require(index == 1240, "native batch plan step count differs");
    std::cout << "SOURCE_NATIVE_BATCH_PLAN_EXACT steps=1240 epochs=40\n";
    neural::RawFBC net(initialized, torch::kCPU);
    auto parameters = net.parameters();
    auto body_before = parameters[0].detach().clone(),
         head_before = parameters[4].detach().clone();
    torch::optim::AdamW opt(
        parameters, torch::optim::AdamWOptions(.001).weight_decay(.001));
    auto schedule = fixture.get("expected_schedule");
    require(schedule.sizes() == torch::IntArrayRef({1240, 2}),
            "full scheduler fixture required");
    std::vector<Tensor> actual_schedule;
    for (int64_t i = 0; i < 1240; ++i) {
      auto [lr, beta] = neural::one_cycle(i, 1240);
      actual_schedule.push_back(torch::tensor({lr, beta}, torch::kFloat64));
    }
    same("OneCycle", torch::stack(actual_schedule), schedule, 1e-15);
    auto ids = torch::tensor(prepared.epochs[0][0], torch::kInt64);
    auto x = prepared.source.x.index_select(0, ids),
         y = prepared.labels.index_select(0, ids);
    auto update = neural::raw_source_update(net, opt, x, y, 0);
    Params observed;
    observed.tensors["native_format"] = torch::tensor(1, torch::kInt64);
    observed.tensors["first_logits"] = update.logits.cpu().clone();
    observed.tensors["first_loss"] = update.loss.cpu().clone();
    std::cout << "native_forward_seconds=" << update.forward_seconds
              << " native_backward_seconds=" << update.backward_seconds
              << " native_optimizer_seconds=" << update.optimizer_seconds
              << '\n';
    same("first_logits", update.logits, fixture.get("first_logits"), 1e-5);
    same("first_posterior",
         torch::softmax(update.logits.to(torch::kFloat64), 1),
         torch::softmax(fixture.get("first_logits").to(torch::kFloat64), 1),
         1e-6);
    same("first_loss", update.loss, fixture.get("first_loss"), 1e-6);
    const std::vector<std::string> names{
        "spatial_conv.0.parametrizations.weight.original",
        "spatial_conv.0.bias",
        "spatial_conv.1.weight",
        "spatial_conv.1.bias",
        "final_layer.parametrizations.weight.original",
        "final_layer.bias"};
    for (size_t i = 0; i < parameters.size(); ++i) {
      require(parameters[i].grad().defined(),
              "native source gradient inactive");
      same("gradient/" + names[i], parameters[i].grad(),
           fixture.get("first_gradient_" + names[i]), 1e-5);
      observed.tensors["gradient_" + names[i]] =
          parameters[i].grad().detach().cpu().clone();
    }
    require(
        (parameters[0].detach() - body_before).abs().max().item<double>() > 0 &&
            (parameters[4].detach() - head_before).abs().max().item<double>() >
                0,
        "native first source body/head optimizer update inactive");
    for (const auto &[name, value] : net.state().tensors) {
      same("state/" + name, value, fixture.get("first_state_" + name), 1e-6);
      observed.tensors["state_" + name] = value;
    }
    observed.save("native_source_update_observed.pt");
    std::cout << "ALL_SOURCE_UPDATE_NUMERIC_FIELDS_COLLECTED" << std::endl;
    if (argc == 4) {
      auto prefix_oracle = Params::load(argv[3]);
      require(prefix_oracle.integer("prefix_steps") == 6,
              "source prefix oracle must contain six optimizer steps");
      compare_prefix_step(prefix_oracle, net, opt, update, names, 1);
      for (int64_t i = 1; i < 6; ++i) {
        auto batch = torch::tensor(prepared.epochs[0][i], torch::kInt64);
        auto next = neural::raw_source_update(
            net, opt, prepared.source.x.index_select(0, batch),
            prepared.labels.index_select(0, batch), i);
        if (i == 1 || i == 5)
          compare_prefix_step(prefix_oracle, net, opt, next, names, i + 1);
      }
      std::cout << "ALL_SOURCE_PREFIX_NUMERIC_FIELDS_COLLECTED" << std::endl;
    }
    require(
        !numeric_failed,
        "NEURAL_FIRST_SOURCE_UPDATE_NUMERIC_FAIL; original bounds exceeded");
    std::cout << "NEURAL_FIRST_SOURCE_UPDATE_PASS maxabs=" << maximum << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
