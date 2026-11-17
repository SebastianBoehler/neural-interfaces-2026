#pragma once
#include "ni26/artifact.h"

namespace ni26 {
// First parity entry consumes an explicit source-only initial state and padded
// source batch schedule. Native source preparation/init/sampling are tracked
// separately; this interface cannot be described as their completed port.
Params train_raw_fbc(const Params &fixture, torch::Device device);
} // namespace ni26
