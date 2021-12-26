#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <kspc/math.hpp>

TEST_CASE("control", "[control]") {
  CHECK(kspc::pi > 0);
}
