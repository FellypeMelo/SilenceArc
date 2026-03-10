#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cstdio>

namespace sa::test {

struct TestResult {
    bool success;
    std::string message;
};

class TestHarness {
public:
    static TestHarness& instance() {
        static TestHarness instance;
        return instance;
    }

    void add_test(const std::string& name, std::function<void()> test_func) {
        m_tests.push_back({name, test_func});
    }

    int run_all() {
        int passed = 0;
        for (const auto& test : m_tests) {
            fprintf(stdout, "[RUN] %s...\n", test.name.c_str());
            fflush(stdout);
            try {
                test.func();
                fprintf(stdout, "[PASS] %s\n", test.name.c_str());
                fflush(stdout);
                passed++;
            } catch (const std::exception& e) {
                fprintf(stderr, "[FAIL] %s: %s\n", test.name.c_str(), e.what());
                fflush(stderr);
            } catch (...) {
                fprintf(stderr, "[FAIL] %s: Unknown error\n", test.name.c_str());
                fflush(stderr);
            }
        }
        fprintf(stdout, "--- Result: %d/%zu tests passed ---\n", passed, m_tests.size());
        fflush(stdout);
        return (passed == m_tests.size()) ? 0 : 1;
    }

private:
    struct TestEntry {
        std::string name;
        std::function<void()> func;
    };
    std::vector<TestEntry> m_tests;
};

#define SA_ASSERT(expr, msg) \
    if (!(expr)) throw std::runtime_error(std::string("Assertion failed: ") + (msg) + " (" + #expr + ") at " + __FILE__ + ":" + std::to_string(__LINE__))

#define SA_EXPECT_TRUE(expr) SA_ASSERT(expr, "Expected true")

} // namespace sa::test
