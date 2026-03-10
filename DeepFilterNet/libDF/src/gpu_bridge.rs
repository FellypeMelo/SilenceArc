use std::ffi::CString;
use std::os::raw::{c_char, c_float};

#[link(name = "silence_arc_infra")]
extern "C" {
    fn sycl_init() -> bool;
    fn sycl_process(input: *const c_float, output: *mut c_float, size: usize);
    fn sycl_get_device_name(buffer: *mut c_char, max_size: usize);
}

pub struct SYCLBridge;

impl SYCLBridge {
    pub fn init() -> bool {
        unsafe { sycl_init() }
    }

    pub fn process(input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len(), "Input and output buffers must have same length");
        unsafe {
            sycl_process(input.as_ptr(), output.as_mut_ptr(), input.len());
        }
    }

    pub fn get_device_name() -> String {
        let mut buffer = vec![0u8; 256];
        unsafe {
            sycl_get_device_name(buffer.as_mut_ptr() as *mut c_char, buffer.len());
        }
        let c_str = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const c_char) };
        c_str.to_string_lossy().into_owned()
    }
}
