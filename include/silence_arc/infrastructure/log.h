#ifndef SILENCE_ARC_INFRASTRUCTURE_LOG_H_
#define SILENCE_ARC_INFRASTRUCTURE_LOG_H_

#include <cstdlib>
#include <iostream>

namespace silence_arc {
namespace infrastructure {

// Verbose diagnostics (per-layer NN build dumps, engine build progress) are
// silenced by default so production stdout stays clean. Set the environment
// variable SILENCEARC_VERBOSE to any non-empty value that does not start with
// '0' to enable them. Errors and warnings are never gated by this.
inline bool verbose_logging_enabled() {
    static const bool enabled = [] {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        const char* v = std::getenv("SILENCEARC_VERBOSE");
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

} // namespace infrastructure
} // namespace silence_arc

// Streamed verbose logging, e.g. SA_LOG_DEBUG("x=" << x); / SA_LOG_INFO("done");
#define SA_LOG_DEBUG(msg)                                                       \
    do {                                                                        \
        if (::silence_arc::infrastructure::verbose_logging_enabled()) {         \
            std::cout << "[DEBUG] " << msg << std::endl;                        \
        }                                                                       \
    } while (0)

#define SA_LOG_INFO(msg)                                                        \
    do {                                                                        \
        if (::silence_arc::infrastructure::verbose_logging_enabled()) {         \
            std::cout << msg << std::endl;                                      \
        }                                                                       \
    } while (0)

#endif // SILENCE_ARC_INFRASTRUCTURE_LOG_H_
