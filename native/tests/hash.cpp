#include "ni26/hash.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    auto check = [](const std::string &message, const std::string &expected) {
      if (ni26::sha256(message.data(), message.size()) != expected)
        throw std::runtime_error("SHA-256 known-answer mismatch");
    };
    check("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    check(std::string(1000000, 'a'),
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    if (ni26::sha256(nullptr, 0) !=
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
      throw std::runtime_error("empty null hash differs");
    bool rejected = false;
    try {
      ni26::sha256(nullptr, 1);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("nonempty null hash accepted");
    if (argc != 2)
      throw std::runtime_error("usage: hash_test REAL_FIR_BYTES");
    std::ifstream file(argv[1], std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot open real FIR fixture");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.size() != 3724 ||
        ni26::sha256(bytes.data(), bytes.size()) !=
            "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6")
      throw std::runtime_error("actual shared FIR fingerprint differs");
    bytes[0] ^= 1;
    if (ni26::sha256(bytes.data(), bytes.size()) ==
        "607a44ee25576803092c03d77feb67411fe7a2266475fc730ae7ae28c81506c6")
      throw std::runtime_error("changed FIR fingerprint accepted");
    std::cout << "SHA256_KNOWN_AND_REAL_FIR_PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
