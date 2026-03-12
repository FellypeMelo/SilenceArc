#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <memory>
#include <map>

namespace sa::infrastructure::sycl_impl {

/**
 * @brief Manages Unified Shared Memory (USM) allocations for SYCL.
 * Ensures proper allocation and deallocation of device, host, and shared memory.
 */
class SyclMemoryManager {
public:
    explicit SyclMemoryManager(sycl::queue& queue);
    ~SyclMemoryManager();

    // Prevent copying
    SyclMemoryManager(const SyclMemoryManager&) = delete;
    SyclMemoryManager& operator=(const SyclMemoryManager&) = delete;

    /**
     * @brief Allocates USM device memory.
     */
    template <typename T>
    T* allocate_device(size_t count) {
        T* ptr = sycl::malloc_device<T>(count, m_queue);
        if (ptr) m_allocations[ptr] = {AllocationType::Device, count * sizeof(T)};
        return ptr;
    }

    /**
     * @brief Allocates USM host memory (pinned).
     */
    template <typename T>
    T* allocate_host(size_t count) {
        T* ptr = sycl::malloc_host<T>(count, m_queue);
        if (ptr) m_allocations[ptr] = {AllocationType::Host, count * sizeof(T)};
        return ptr;
    }

    /**
     * @brief Allocates USM shared memory (accessible by both CPU and GPU).
     */
    template <typename T>
    T* allocate_shared(size_t count) {
        T* ptr = sycl::malloc_shared<T>(count, m_queue);
        if (ptr) m_allocations[ptr] = {AllocationType::Shared, count * sizeof(T)};
        return ptr;
    }

    /**
     * @brief Frees an allocated USM pointer.
     */
    void free(void* ptr);

    /**
     * @brief Frees all managed allocations.
     */
    void free_all();

private:
    enum class AllocationType { Device, Host, Shared };
    struct AllocationInfo {
        AllocationType type;
        size_t size;
    };

    sycl::queue& m_queue;
    std::map<void*, AllocationInfo> m_allocations;
};

} // namespace sa::infrastructure::sycl_impl
