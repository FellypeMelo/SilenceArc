use std::os::raw::{c_char, c_float};

pub struct SYCLBridge;

impl SYCLBridge {
    pub fn init() -> bool {
        false
    }

    pub fn process(_input: &[f32], output: &mut [f32]) {
        output.fill(0.0);
    }

    pub fn get_device_name() -> String {
        "CPU (Fallback)".to_string()
    }
}

// These stubs are only here to satisfy the linker if something else expects them,
// but they should NOT be called if we are not linking against the real SYCL implementation.
#[no_mangle]
pub extern "C" fn sycl_init() -> bool { false }
#[no_mangle]
pub extern "C" fn sycl_process(_input: *const c_float, _output: *mut c_float, _size: usize) {}
#[no_mangle]
pub extern "C" fn sycl_get_device_name(_buffer: *mut c_char, _max_size: usize) {}
