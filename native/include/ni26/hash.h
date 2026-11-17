#pragma once
#include <cstddef>
#include <string>

namespace ni26 {
// SHA-256 for immutable artifact fingerprints. No mutable global hash state.
std::string sha256(const void *data, std::size_t size);
} // namespace ni26
