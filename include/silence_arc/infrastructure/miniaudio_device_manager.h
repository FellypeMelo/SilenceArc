#ifndef SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_DEVICE_MANAGER_H_
#define SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_DEVICE_MANAGER_H_

#include "silence_arc/domain/ui_state.h"

namespace silence_arc {
namespace infrastructure {

class MiniaudioDeviceManager {
public:
    static bool EnumerateDevices(domain::UIState& state);
};

} // namespace infrastructure
} // namespace silence_arc

#endif // SILENCE_ARC_INFRASTRUCTURE_MINIAUDIO_DEVICE_MANAGER_H_
