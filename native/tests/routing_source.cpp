#include "ni26/output.h"
#include "ni26/routing.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
using namespace ni26;
int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: routing_source PARAMS FIXTURE");
    torch::set_num_threads(1);
    auto z = Params::load(argv[1]), f = Params::load(argv[2]);
    double maximum = 0;
    std::map<std::string, double> errors;
    for (const std::string mode : {"known", "unknown"}) {
      Route route(z, z.get("member_weights")[0].item<double>() /
                         z.get("member_weights").sum().item<double>());
      CountFilter cf(z);
      CausalPers cp(z);
      auto X = f.get(mode + "_x");
      for (int64_t i = 0; i < X.size(0); i++) {
        auto read = [&](const std::string &key) {
          return f.get(mode + "_" + key)[i];
        };
        Tensor routed;
        if (mode == "known")
          routed = route.step(X[i], read("u"), read("mix"), read("core1"),
                              read("shadow"));
        else
          routed = route.step_given(X[i], read("u"), read("mix"), read("core1"),
                                    {}, {}, {});
        auto count = cf.step(X[i], read("p"));
        auto personal = cp.step(X[i], read("p"), count,
                                read("recognized").item<int64_t>() != 0);
        int field = 0;
        for (auto pair : {std::pair<Tensor, Tensor>{routed, read("route")},
                          {count, read("count")},
                          {personal, read("cp")}}) {
          double error = (pair.first - pair.second).abs().max().item<double>();
          maximum = std::max(maximum, error);
          std::string key =
              mode + "/" +
              std::vector<std::string>{"route", "count", "cp"}[field++];
          errors[key] = std::max(errors[key], error);
          if (error > 1e-8)
            std::cerr << key << " window=" << i
                      << " maxabs=" << std::setprecision(17) << error
                      << std::endl;
          auto sorted = std::get<0>(pair.second.sort());
          if ((sorted[-1] - sorted[-2]).item<double>() > 2e-8)
            require(pair.first.argmax().item<int64_t>() ==
                        pair.second.argmax().item<int64_t>(),
                    "non-tied source class changed");
          require(error <= 1e-8, "source stage mismatch at " + mode + "/" +
                                     std::to_string(i) +
                                     " maxabs=" + std::to_string(error));
        }
        if (mode == "known")
          for (auto pair : {std::pair<Decision, Tensor>{route.last_contrast(),
                                                        read("contrast")},
                            {route.last_identity(), read("identity")}}) {
            auto expected = pair.second;
            if (expected[0].item<int64_t>() < 0)
              require(!pair.first, "unexpected gate decision");
            else {
              require(bool(pair.first), "missing gate decision");
              require(pair.first->person == expected[0].item<int64_t>(),
                      "gate person changed");
              require(std::abs(pair.first->score -
                               expected[1].item<double>()) <= 1e-8 &&
                          std::abs(pair.first->weight -
                                   expected[2].item<double>()) <= 1e-8,
                      "gate score/weight changed");
            }
          }
        if ((i == 16 || i == 203) && i + 1 < X.size(0)) {
          auto next = [&](const std::string &key) {
            return f.get(mode + "_" + key)[i + 1];
          };
          Route a = route, b = route;
          CountFilter ca = cf, cb = cf;
          CausalPers pa = cp, pb = cp;
          auto ra = mode == "known"
                        ? a.step(X[i + 1], next("u"), next("mix"),
                                 next("core1"), next("shadow"))
                        : a.step_given(X[i + 1], next("u"), next("mix"),
                                       next("core1"), {}, {}, {});
          auto rb = mode == "known"
                        ? b.step(X[i + 1], next("u"), next("mix"),
                                 next("core1"), next("shadow"))
                        : b.step_given(X[i + 1], next("u"), next("mix"),
                                       next("core1"), {}, {}, {});
          require(torch::equal(ra, rb), "route deepcopy differs");
          auto oa = ca.step(X[i + 1], next("p")),
               ob = cb.step(X[i + 1], next("p"));
          require(torch::equal(oa, ob), "count deepcopy differs");
          require(
              torch::equal(pa.step(X[i + 1], next("p"), oa,
                                   next("recognized").item<int64_t>() != 0),
                           pb.step(X[i + 1], next("p"), ob,
                                   next("recognized").item<int64_t>() != 0)),
              "CP deepcopy differs");
        }
      }
      require(!cp.stats().dead && !cf.stats().dead,
              "silent native stage bypass");
      if (mode == "unknown")
        require(cp.stats().refits ==
                        (214 - z.integer("v6_cp_t0")) / z.integer("v6_cp_k") +
                            1 &&
                    cp.stats().fused == 125,
                "native late CP inactive");
      route.reset();
      cf.reset();
      cp.reset();
      require(route.stats().windows == 0 && cf.stats().windows == 0 &&
                  cp.stats().windows == 0,
              "reset counters nonzero");
      for (int64_t i = 0; i < 17; i++) {
        auto read = [&](const std::string &key) {
          return f.get(mode + "_" + key)[i];
        };
        auto routed = mode == "known"
                          ? route.step(X[i], read("u"), read("mix"),
                                       read("core1"), read("shadow"))
                          : route.step_given(X[i], read("u"), read("mix"),
                                             read("core1"), {}, {}, {});
        auto count = cf.step(X[i], read("p"));
        auto personal = cp.step(X[i], read("p"), count,
                                read("recognized").item<int64_t>() != 0);
        for (auto pair : {std::pair<Tensor, Tensor>{routed, read("route")},
                          {count, read("count")},
                          {personal, read("cp")}})
          require((pair.first - pair.second).abs().max().item<double>() <= 1e-8,
                  "reset source prefix differs");
      }

      std::cout << mode << " SOURCE_PASS maxabs=" << maximum << std::endl;
    }
    for (auto &item : errors)
      std::cout << item.first << " maxabs=" << std::setprecision(17)
                << item.second << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
