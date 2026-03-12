#include "silence_arc/infrastructure/sycl/sycl_memory_manager.h"
#include <iostream>

namespace sa::infrastructure::sycl_impl {

SyclMemoryManager::SyclMemoryManager(sycl::queue& queue) : m_queue(queue) {}

SyclMemoryManager::~SyclMemoryManager() {
    free_all();
}

void SyclMemoryManager::free(void* ptr) {
    if (!ptr) return;

    auto it = m_allocations.find(ptr);
    if (it != m_allocations.end()) {
        sycl::free(ptr, m_queue);
        m_allocations.erase(it);
    }
}

void SyclMemoryManager::free_all() {
    for (auto const& [ptr, info] : m_allocations) {
        sycl::free(ptr, m_queue);
    }
    m_allocations.clear();
}

} // namespace sa::infrastructure::sycl_impl
