#pragma once

#include "silence_arc/infrastructure/ui_manager.h"
#include <vector>
#include <string>

namespace sa::infrastructure {

class MiniaudioDeviceManager {
public:
    static void EnumerateDevices(UIState& state);
};

} // namespace sa::infrastructure
