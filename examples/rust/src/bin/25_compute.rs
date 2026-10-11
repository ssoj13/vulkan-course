//! Step 25 - Compute shaders.
//!
//! No window, no rasterizer: a COMPUTE PIPELINE is one shader stage, and
//! vkCmdDispatch launches a grid of workgroups that run it. Two kernels:
//!
//!   saxpy  (shaders/saxpy.slang):  result[i] = a * x[i] + y[i], one invocation per element
//!   reduce (shaders/reduce.slang): every workgroup sums its 256 values in groupshared memory
//!
//!   buffers (STORAGE_BUFFER) -> descriptor set per kernel -> compute pipeline
//!   record: push constants, dispatch ceil(N / 256) groups, barrier COMPUTE -> HOST
//!   submit, wait for the fence, read the results through mapped HOST_CACHED memory
//!
//! The CPU computes the same results and compares them exactly (integers in
//! float and uint). `--count N` sets the number of elements (default
//! 1000000, deliberately not a multiple of 256), `--uncached` reads the
//! results from uncached memory to compare the CPU read time. Lines that
//! depend on timing start with `[timing]`. Steps 01-05 are repeated, reduced
//! to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::time::Instant;

use ash::{
    Entry,
    ext::debug_utils,
    vk::{self, TaggedStructure as _},
};

/// Validation messages are printed here (explained in step 02).
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "ERROR"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        "WARNING"
    } else {
        "INFO"
    };
    let kind = if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "VALIDATION"
    } else if msg_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "PERFORMANCE"
    } else {
        "GENERAL"
    };
    // SAFETY: the layer passes a valid pointer for the duration of the call
    // (we still guard against null); `p_message` is a NUL-terminated string.
    let message = unsafe { data.as_ref() }
        .filter(|d| !d.p_message.is_null())
        .map(|d| unsafe { CStr::from_ptr(d.p_message) }.to_string_lossy())
        .unwrap_or(std::borrow::Cow::Borrowed("<no message>"));
    eprintln!("[{level}][{kind}] {message}");
    vk::FALSE
}

// #region cli
/// Command line: `--count N` elements (default 1000000) and `--uncached`.
struct Args {
    count: u32,
    uncached: bool,
}

/// The largest `--count`: saxpy's results go up to 2 * count - 1, and every
/// integer up to 2^24 is exact in a float, so count <= 2^23 keeps the GPU's
/// float results exactly comparable with the CPU's.
const MAX_COUNT: u32 = 1 << 23;

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let usage = "usage: 25_compute [--count N] [--uncached]";
        let mut args = Args {
            count: 1_000_000,
            uncached: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--count" => args.count = it.next().ok_or(usage)?.parse()?,
                "--uncached" => args.uncached = true,
                _ => return Err(usage.into()),
            }
        }
        if args.count == 0 || args.count > MAX_COUNT {
            return Err(format!("--count must be 1-{MAX_COUNT}").into());
        }
        Ok(args)
    }
}
// #endregion

/// The scale of saxpy.
const A: f32 = 2.0;

/// Memory property names for printing, in bit order (step 13).
fn memory_names(flags: vk::MemoryPropertyFlags) -> String {
    let table = [
        (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
    ];
    let parts: Vec<&str> = table
        .iter()
        .filter(|(bit, _)| flags.contains(*bit))
        .map(|&(_, name)| name)
        .collect();
    if parts.is_empty() {
        "(none)".to_owned()
    } else {
        parts.join(" | ")
    }
}

/// Step 13's find-memory-type rule: the first memory type, in index order,
/// that `bits` (the resource's memoryTypeBits) allows and that has every flag.
fn find_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count)
        .zip(props.memory_types_as_slice())
        .find_map(|(i, t)| (bits & (1 << i) != 0 && t.property_flags.contains(flags)).then_some(i))
}

/// Step 13's two passes: `required | preferred` first, then `required` alone.
fn choose_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<u32> {
    find_type(props, bits, required | preferred).or_else(|| find_type(props, bits, required))
}

/// The memory types that have any of `flags`, as a memoryTypeBits-style
/// mask: `bits & !types_with(..)` rules them out of a choice.
fn types_with(props: &vk::PhysicalDeviceMemoryProperties, flags: vk::MemoryPropertyFlags) -> u32 {
    (0..props.memory_type_count)
        .zip(props.memory_types_as_slice())
        .filter(|(_, t)| t.property_flags.intersects(flags))
        .fold(0, |mask, (i, _)| mask | (1 << i))
}

// #region local-size
/// The workgroup size of `entry`, read from the SPIR-V: the module's
/// OpEntryPoint names the function, OpExecutionMode LocalSize gives x, y, z.
/// The host needs it to turn "N elements" into "groups to dispatch"; reading
/// it from the module keeps one source of truth (the shader's [numthreads]).
fn local_size(words: &[u32], entry: &str) -> Result<[u32; 3], Box<dyn Error>> {
    const OP_ENTRY_POINT: u32 = 15;
    const OP_EXECUTION_MODE: u32 = 16;
    const LOCAL_SIZE: u32 = 17;
    let mut function = None;
    let mut sizes = Vec::new();
    // Instructions start after the 5-word header; each one's first word is
    // (word count << 16) | opcode.
    let mut pos = 5;
    while let Some(&first) = words.get(pos) {
        let count = usize::try_from(first >> 16)?;
        let ins = words
            .get(pos..pos + count)
            .filter(|_| count > 0)
            .ok_or("malformed SPIR-V instruction")?;
        match (first & 0xffff, ins) {
            // Operands: execution model, function id, name (a literal string), interface ids.
            (OP_ENTRY_POINT, [_, _model, id, name @ ..]) => {
                let bytes: Vec<u8> = name
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .collect();
                if bytes == entry.as_bytes() {
                    function = Some(*id);
                }
            }
            (OP_EXECUTION_MODE, [_, id, LOCAL_SIZE, x, y, z]) => sizes.push((*id, [*x, *y, *z])),
            _ => {}
        }
        pos += count;
    }
    let function = function.ok_or(format!("no entry point {entry}"))?;
    sizes
        .iter()
        .find(|(id, _)| *id == function)
        .map(|&(_, size)| size)
        .ok_or_else(|| format!("{entry} has no LocalSize execution mode").into())
}
// #endregion

/// A buffer, its memory and the memory's persistent mapping.
#[derive(Clone, Copy)]
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_type: u32,
    ptr: *mut c_void,
}

/// A compute kernel: its descriptor set layout, pipeline layout and pipeline.
#[derive(Clone, Copy)]
struct Kernel {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// Every device object `run` creates. Dropping it waits for the device to go
/// idle and destroys them in reverse order of creation, so an early return
/// through `?` cleans up like the normal path.
struct Objects<'a> {
    device: &'a ash::Device,
    buffers: Vec<GpuBuffer>,
    kernels: Vec<Kernel>,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
}

impl Drop for Objects<'_> {
    fn drop(&mut self) {
        let device = self.device;
        // SAFETY: plain call on a live device.
        if let Err(e) = unsafe { device.device_wait_idle() } {
            eprintln!("device_wait_idle failed: {e}");
        }
        // SAFETY: the device is idle, every handle was created from it and is
        // destroyed exactly once (destroying a null handle is a no-op);
        // freeing a mapped allocation unmaps it.
        unsafe {
            device.destroy_query_pool(self.query_pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for k in self.kernels.iter().rev() {
                device.destroy_pipeline(k.pipeline, None);
                device.destroy_pipeline_layout(k.layout, None);
                device.destroy_descriptor_set_layout(k.set_layout, None);
            }
            for b in self.buffers.iter().rev() {
                device.destroy_buffer(b.buffer, None);
                device.free_memory(b.memory, None);
            }
        }
    }
}

/// Creates a STORAGE_BUFFER of `size` bytes in the type `choose_type` picks
/// among those without any `excluded` flag, and maps it for the program's
/// lifetime. The buffer is added to `objects` as soon as it exists, so an
/// error halfway still destroys it.
fn create_buffer(
    objects: &mut Objects,
    props: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
    excluded: vk::MemoryPropertyFlags,
) -> Result<GpuBuffer, Box<dyn Error>> {
    let device = objects.device;
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: plain create call on a live device.
    let buffer = unsafe { device.create_buffer(&info, None)? };
    objects.buffers.push(GpuBuffer {
        buffer,
        memory: vk::DeviceMemory::null(),
        memory_type: 0,
        ptr: std::ptr::null_mut(),
    });
    let index = objects.buffers.len() - 1;
    // SAFETY: `buffer` belongs to this device.
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let bits = req.memory_type_bits & !types_with(props, excluded);
    let memory_type = choose_type(props, bits, required, preferred).ok_or_else(|| {
        format!(
            "no memory type with {} and without {}",
            memory_names(required),
            memory_names(excluded)
        )
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type);
    // SAFETY: the type index comes from this device's memory properties.
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    let slot = objects
        .buffers
        .get_mut(index)
        .ok_or("buffer list out of sync")?;
    slot.memory = memory;
    slot.memory_type = memory_type;
    // SAFETY: the allocation has the required size and type; offset 0 is
    // aligned. The memory is HOST_VISIBLE (required) and not mapped yet.
    unsafe {
        device.bind_buffer_memory(buffer, memory, 0)?;
        slot.ptr = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
    }
    Ok(*slot)
}

// #region kernel
/// One compute kernel: a set layout with `bindings` storage buffers (binding
/// 0, 1, ...), a pipeline layout with that set and a push constant range of
/// `push_size` bytes, and a compute pipeline running `entry` of `spirv`.
/// Compared with a graphics pipeline there is nothing else to describe: no
/// vertex input, no rasterizer, no attachments - one stage.
fn create_kernel(
    objects: &mut Objects,
    spirv: &[u32],
    entry: &CStr,
    bindings: u32,
    push_size: u32,
) -> Result<Kernel, Box<dyn Error>> {
    let device = objects.device;
    let set_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings);
    objects.kernels.push(Kernel {
        set_layout: vk::DescriptorSetLayout::null(),
        layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
    });
    let index = objects.kernels.len() - 1;
    let kernel = objects
        .kernels
        .get_mut(index)
        .ok_or("kernel list out of sync")?;
    // SAFETY: plain create call on a live device.
    kernel.set_layout = unsafe { device.create_descriptor_set_layout(&set_info, None)? };

    let set_layouts = [kernel.set_layout];
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_size)];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&ranges);
    // SAFETY: the set layout belongs to this device.
    kernel.layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let module_info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: `spirv` is valid SPIR-V (checked by spirv-val in sync:shaders).
    let module = unsafe { device.create_shader_module(&module_info, None)? };
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry),
        )
        .layout(kernel.layout);
    // SAFETY: all create-info data outlives the call; no pipeline cache.
    let created = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)
    };
    // SAFETY: the module is used by no other pending create call; a pipeline
    // keeps working after its module is destroyed.
    unsafe { device.destroy_shader_module(module, None) };
    kernel.pipeline = created?
        .first()
        .copied()
        .ok_or("vkCreateComputePipelines returned no pipeline")?;
    Ok(*kernel)
}
// #endregion

/// Everything after device creation. `objects` owns what it creates, so any
/// error returns through its Drop.
fn run(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    family: u32,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let mut objects = Objects {
        device,
        buffers: Vec::new(),
        kernels: Vec::new(),
        descriptor_pool: vk::DescriptorPool::null(),
        command_pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
        query_pool: vk::QueryPool::null(),
    };

    // #region limits
    // The limits a dispatch must respect, and the subgroup size: how many
    // invocations the hardware packs into one SIMD group (a warp on NVIDIA;
    // the spec promises no lockstep execution). A workgroup is split into
    // subgroups, so its size is best a multiple of this. The subgroup size
    // is a Vulkan 1.1 property, read through the properties2 chain.
    let mut props11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut props11);
    // SAFETY: `physical_device` belongs to `instance`; the chain outlives the call.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
    let limits = props2.properties.limits;
    let subgroup = props11.subgroup_size;
    let [gx, gy, gz] = limits.max_compute_work_group_count;
    let [sx, sy, sz] = limits.max_compute_work_group_size;
    println!("\nCompute limits:");
    println!(
        "  maxComputeWorkGroupInvocations = {} (Vulkan 1.4 guarantees 256)",
        limits.max_compute_work_group_invocations
    );
    println!("  maxComputeWorkGroupSize = {sx} x {sy} x {sz}");
    println!("  maxComputeWorkGroupCount = {gx} x {gy} x {gz}");
    println!(
        "  maxComputeSharedMemorySize = {} bytes",
        limits.max_compute_shared_memory_size
    );
    println!("  subgroupSize = {subgroup}");
    // #endregion

    // The SPIR-V is compiled from shaders/{saxpy,reduce}.slang by `bun run
    // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
    const SAXPY_SPV: &[u8] = include_bytes!("../../../../shaders/spv/saxpy.spv");
    const REDUCE_SPV: &[u8] = include_bytes!("../../../../shaders/spv/reduce.spv");
    let saxpy_spv = ash::util::read_spv(&mut std::io::Cursor::new(SAXPY_SPV))?;
    let reduce_spv = ash::util::read_spv(&mut std::io::Cursor::new(REDUCE_SPV))?;

    // #region group-count
    // The workgroup size comes from the shader; the host only decides how
    // many groups: ceil(count / group size), so the last group may be
    // partly past the end (the shaders check `i < count`). Two kinds of
    // limits apply. The workgroup size is a rule on the shader the pipeline
    // is created with: LocalSize x at most maxComputeWorkGroupSize[0]
    // (VUID-RuntimeSpirv-x-06429), x * y * z at most
    // maxComputeWorkGroupInvocations (VUID-RuntimeSpirv-x-06432). The group
    // count is a rule on vkCmdDispatch: at most maxComputeWorkGroupCount
    // (VUID-vkCmdDispatch-groupCountX-00386).
    println!("\nKernels (workgroup size read from the SPIR-V):");
    let mut sizes = Vec::new();
    for (file, words, entry) in [
        ("saxpy.spv", &saxpy_spv, "saxpy"),
        ("reduce.spv", &reduce_spv, "reduce"),
    ] {
        let [x, y, z] = local_size(words, entry)?;
        // Both kernels are 1D (y = z = 1), so x alone is the invocation count.
        if y != 1 || z != 1 {
            return Err(format!("{entry}: expected a 1D workgroup, got {x}x{y}x{z}").into());
        }
        if x > sx || x > limits.max_compute_work_group_invocations {
            return Err(format!("{entry}: a workgroup of {x} exceeds this device's limits").into());
        }
        println!(
            "  {file}: {} bytes, entry point {entry}, LocalSize {x} x {y} x {z} = {} subgroups of {subgroup}",
            words.len() * 4,
            x.div_ceil(subgroup)
        );
        sizes.push(x);
    }
    let [saxpy_size, reduce_size] = sizes[..] else {
        return Err("expected two workgroup sizes".into());
    };
    // reduce.slang's groupshared array has one uint per invocation.
    let shared_bytes = reduce_size * 4;
    println!(
        "  reduce: groupshared {shared_bytes} bytes per workgroup (limit {})",
        limits.max_compute_shared_memory_size
    );
    let saxpy_groups = args.count.div_ceil(saxpy_size);
    let reduce_groups = args.count.div_ceil(reduce_size);
    if saxpy_groups.max(reduce_groups) > gx {
        return Err(
            format!("{saxpy_groups} groups exceed maxComputeWorkGroupCount[0] = {gx}").into(),
        );
    }
    let tail = args.count - (saxpy_groups - 1) * saxpy_size;
    println!(
        "\n{} elements -> {saxpy_groups} workgroups of {saxpy_size} (the last one has {tail} elements in range)",
        args.count
    );
    // #endregion

    // #region memory
    // Inputs: the CPU writes them once, the GPU reads them: HOST_VISIBLE |
    // HOST_COHERENT, the upload type of step 14 (write-combined: fast to
    // write, slow to read). Outputs: the GPU writes them, the CPU READS
    // them: the spec expects HOST_CACHED for that ("when the host performs
    // any read or read-modify-write operations on the memory"), because
    // "host memory read accesses to uncached memory are slower than to cached
    // memory". Both live in system RAM on this GPU, so the shaders read and
    // write them across PCIe - fine for one pass over the data; data the GPU
    // reuses belongs in DEVICE_LOCAL memory (step 15's staging).
    // `--uncached` rules HOST_CACHED types out instead, for the comparison;
    // the spec adds that "uncached memory is always host coherent".
    // SAFETY: `physical_device` belongs to `instance`.
    let mem = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let none = vk::MemoryPropertyFlags::empty();
    let upload = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let readback_required = vk::MemoryPropertyFlags::HOST_VISIBLE;
    let (readback_preferred, readback_excluded) = if args.uncached {
        (none, vk::MemoryPropertyFlags::HOST_CACHED)
    } else {
        (vk::MemoryPropertyFlags::HOST_CACHED, none)
    };
    let n = usize::try_from(args.count)?;
    let float_bytes = vk::DeviceSize::try_from(n * size_of::<f32>())?;
    let uint_bytes = vk::DeviceSize::try_from(n * size_of::<u32>())?;
    let sums_bytes = vk::DeviceSize::from(reduce_groups) * 4;
    let x = create_buffer(&mut objects, &mem, float_bytes, upload, none, none)?;
    let y = create_buffer(&mut objects, &mem, float_bytes, upload, none, none)?;
    let values = create_buffer(&mut objects, &mem, uint_bytes, upload, none, none)?;
    let result = create_buffer(
        &mut objects,
        &mem,
        float_bytes,
        readback_required,
        readback_preferred,
        readback_excluded,
    )?;
    let sums = create_buffer(
        &mut objects,
        &mem,
        sums_bytes,
        readback_required,
        readback_preferred,
        readback_excluded,
    )?;
    // "The memoryTypeBits member is identical for all VkBuffer objects
    // created with the same value for the flags and usage members": the two
    // outputs get the same type from the same choice, and everything below
    // (the printout, the invalidate) treats them as one.
    if sums.memory_type != result.memory_type {
        return Err("result and groupSums got different memory types".into());
    }
    let type_flags = |t: u32| -> Result<vk::MemoryPropertyFlags, Box<dyn Error>> {
        Ok(mem
            .memory_types_as_slice()
            .get(usize::try_from(t)?)
            .ok_or("memory type out of range")?
            .property_flags)
    };
    let (in_type, out_type) = (x.memory_type, result.memory_type);
    let out_flags = type_flags(out_type)?;
    println!("\nMemory:");
    println!(
        "  inputs x, y, values ({} bytes each): memory type {in_type} ({}), written by the CPU",
        float_bytes,
        memory_names(type_flags(in_type)?)
    );
    println!(
        "  outputs result ({float_bytes} bytes), groupSums ({sums_bytes} bytes): memory type {out_type} ({}), read by the CPU",
        memory_names(out_flags)
    );
    // #endregion

    // #region fill-inputs
    // Integer inputs make every result exact: x[i] = i, y[i] = 1 gives
    // result[i] = 2i + 1 < 2^24; Vulkan requires OpFMul and OpFAdd to be
    // correctly rounded, and an exact result needs no rounding, so the GPU
    // must produce exactly what the CPU does (fused or not).
    let xs: Vec<f32> = (0..args.count).map(|i| i as f32).collect();
    let ys = vec![1.0_f32; n];
    let vs: Vec<u32> = (0..args.count).map(|i| i % 1000).collect();
    for (b, data) in [
        (x, bytemuck_f32(&xs)),
        (y, bytemuck_f32(&ys)),
        (values, bytemuck_u32(&vs)),
    ] {
        // SAFETY: `ptr` maps the whole allocation, which is at least
        // `data.len()` bytes; the GPU does not use it yet. HOST_COHERENT:
        // the submit makes the writes visible to the device (no flush).
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr.cast::<u8>(), data.len()) };
    }
    // #endregion

    // #region descriptors
    // One set per kernel; each binding points at a whole buffer.
    let saxpy = create_kernel(&mut objects, &saxpy_spv, c"saxpy", 3, 8)?;
    let reduce = create_kernel(&mut objects, &reduce_spv, c"reduce", 2, 4)?;
    println!(
        "\nCompute pipelines: saxpy (3 storage buffers, 8 push constant bytes), reduce (2 storage buffers, 4 bytes)"
    );
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(5)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2)
        .pool_sizes(&pool_sizes);
    // SAFETY: plain create call on a live device.
    objects.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    let set_layouts = [saxpy.set_layout, reduce.set_layout];
    let alloc_sets = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(objects.descriptor_pool)
        .set_layouts(&set_layouts);
    // SAFETY: the pool and the layouts belong to this device; the pool has room.
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_sets)? };
    let [saxpy_set, reduce_set] = sets[..] else {
        return Err("expected two descriptor sets".into());
    };
    let infos = [
        (saxpy_set, 0, x),
        (saxpy_set, 1, y),
        (saxpy_set, 2, result),
        (reduce_set, 0, values),
        (reduce_set, 1, sums),
    ]
    .map(|(set, binding, b)| {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(b.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        (set, binding, info)
    });
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .map(|(set, binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(*binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        })
        .collect();
    // SAFETY: every set, buffer and info outlives the call; no command buffer uses the sets yet.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    println!("Descriptor sets: saxpy x/y/result at bindings 0/1/2, reduce values/groupSums at 0/1");
    // #endregion

    // #region timestamps
    // Three GPU timestamps around the dispatches (step 12 explains the query
    // pool, timestampValidBits and timestampPeriod).
    // SAFETY: `physical_device` belongs to `instance`.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let valid_bits = families
        .get(usize::try_from(family)?)
        .ok_or("queue family out of range")?
        .timestamp_valid_bits;
    if valid_bits == 0 {
        return Err("The compute queue family has no timestamps".into());
    }
    let query_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(3);
    // SAFETY: plain create call on a live device.
    objects.query_pool = unsafe { device.create_query_pool(&query_info, None)? };
    // #endregion

    // One command buffer, submitted once (step 09).
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
    // SAFETY: plain create calls on a live device.
    unsafe {
        objects.command_pool = device.create_command_pool(&pool_info, None)?;
        objects.fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
    }
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(objects.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool belongs to this device.
    let cmd = *unsafe { device.allocate_command_buffers(&alloc_info)? }
        .first()
        .ok_or("no command buffer")?;

    // #region record
    // The compute bind point has its own pipeline, descriptor sets and push
    // constants, separate from the graphics ones. The two dispatches touch
    // different buffers, so nothing orders them: the GPU may run them
    // concurrently. Timestamp 0 at TOP_OF_PIPE, 1 after saxpy's COMPUTE_SHADER
    // work, 2 after both.
    let push_saxpy = [A.to_ne_bytes(), args.count.to_ne_bytes()].concat();
    let push_reduce = args.count.to_ne_bytes();
    use vk::PipelineStageFlags2 as S;
    // SAFETY: `cmd` comes from a pool of this device; every handle is alive
    // until the fence below has signaled.
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_reset_query_pool(cmd, objects.query_pool, 0, 3);
        device.cmd_write_timestamp2(cmd, S::TOP_OF_PIPE, objects.query_pool, 0);

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, saxpy.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            saxpy.layout,
            0,
            &[saxpy_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            saxpy.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_saxpy,
        );
        device.cmd_dispatch(cmd, saxpy_groups, 1, 1);
        device.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, objects.query_pool, 1);

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            reduce.layout,
            0,
            &[reduce_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            reduce.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_reduce,
        );
        device.cmd_dispatch(cmd, reduce_groups, 1, 1);
        device.cmd_write_timestamp2(cmd, S::COMPUTE_SHADER, objects.query_pool, 2);
    }
    println!(
        "\nRecorded: dispatch saxpy {saxpy_groups} x 1 x 1, dispatch reduce {reduce_groups} x 1 x 1"
    );
    // #endregion

    // #region readback-barrier
    // The CPU will read what the shaders wrote. The fence alone is not
    // enough: "Signaling a fence and waiting on the host does not guarantee
    // that the results of memory accesses will be visible to the host, as
    // the access scope of a memory dependency defined by a fence only
    // includes device access." This barrier makes the shader writes
    // available and visible to the HOST domain: source = the compute stage's
    // storage writes, destination = host reads.
    let to_host = vk::MemoryBarrier2::default()
        .src_stage_mask(S::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(S::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ);
    // SAFETY: `cmd` is recording.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&to_host)),
        );
        device.end_command_buffer(cmd)?;
    }
    println!("Barrier: COMPUTE_SHADER / SHADER_STORAGE_WRITE -> HOST / HOST_READ");
    // #endregion

    // #region submit-and-wait
    let cmd_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let submit = vk::SubmitInfo2::default().command_buffer_infos(std::slice::from_ref(&cmd_info));
    // SAFETY: one queue of `family` was created with the device.
    let queue = unsafe { device.get_device_queue(family, 0) };
    // SAFETY: all handles belong to this device; the fence is unsignaled.
    unsafe {
        device.queue_submit2(queue, &[submit], objects.fence)?;
        device.wait_for_fences(&[objects.fence], true, u64::MAX)?;
    }
    let mut ticks = [0_u64; 3];
    // SAFETY: the queries were written by the completed submission.
    unsafe {
        device.get_query_pool_results(
            objects.query_pool,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
    }
    let mask = u64::MAX >> (64 - valid_bits.min(64));
    let period = f64::from(limits.timestamp_period);
    let us = |from: u64, to: u64| (to.wrapping_sub(from) & mask) as f64 * period / 1000.0;
    let [t0, t1, t2] = ticks;
    // saxpy moves 12 bytes per element: reads x and y, writes result.
    let saxpy_us = us(t0, t1);
    println!(
        "[timing] GPU: saxpy {saxpy_us:.1} us ({:.1} GB/s over PCIe), saxpy + reduce {:.1} us",
        (n * 12) as f64 / (saxpy_us * 1e3),
        us(t0, t2)
    );
    // #endregion

    // #region read-results
    // A non-coherent type would need vkInvalidateMappedMemoryRanges here, so
    // the CPU does not read stale cache lines; HOST_COHERENT types do not.
    if !out_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        let ranges = [result, sums].map(|b| {
            vk::MappedMemoryRange::default()
                .memory(b.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)
        });
        // SAFETY: both allocations are mapped.
        unsafe { device.invalidate_mapped_memory_ranges(&ranges)? };
        println!(
            "Output memory is not HOST_COHERENT: vkInvalidateMappedMemoryRanges before reading"
        );
    } else {
        println!("Output memory is HOST_COHERENT: no vkInvalidateMappedMemoryRanges needed");
    }
    let start = Instant::now();
    let groups = usize::try_from(reduce_groups)?;
    // SAFETY: both mappings hold at least this many elements, written by the
    // completed submission and made visible to the host by the barrier.
    let (gpu_result, gpu_sums) = unsafe {
        (
            std::slice::from_raw_parts(result.ptr.cast::<f32>(), n).to_vec(),
            std::slice::from_raw_parts(sums.ptr.cast::<u32>(), groups).to_vec(),
        )
    };
    let read = start.elapsed();
    println!(
        "[timing] CPU copy of the results out of memory type {out_type}: {:.2} ms ({:.2} GB/s)",
        read.as_secs_f64() * 1e3,
        (n * 4 + groups * 4) as f64 / read.as_secs_f64() / 1e9
    );
    // #endregion

    // #region verify-saxpy
    let mismatches = xs
        .iter()
        .zip(&ys)
        .zip(&gpu_result)
        .filter(|&((&xi, &yi), &gi)| A * xi + yi != gi)
        .count();
    let last = n - 1;
    println!("\nsaxpy: result[i] = {A} * x[i] + y[i], x[i] = i, y[i] = 1");
    let first: Vec<String> = gpu_result.iter().take(4).map(f32::to_string).collect();
    println!(
        "  result[0..{}] = {}, result[{last}] = {}",
        first.len(),
        first.join(", "),
        gpu_result.get(last).copied().unwrap_or_default()
    );
    if mismatches != 0 {
        return Err(format!("saxpy: {mismatches} of {n} results differ from the CPU").into());
    }
    println!("  all {n} results equal the CPU's exactly");
    // #endregion

    // #region verify-reduce
    let cpu_sums: Vec<u32> = vs
        .chunks(usize::try_from(reduce_size)?)
        .map(|c| c.iter().sum())
        .collect();
    let cpu_total: u64 = vs.iter().map(|&v| u64::from(v)).sum();
    let gpu_total: u64 = gpu_sums.iter().map(|&v| u64::from(v)).sum();
    println!("\nreduce: values[i] = i % 1000, one sum per workgroup of {reduce_size}");
    println!(
        "  groupSums[0] = {}, groupSums[{}] = {} (the last group has {} values)",
        gpu_sums.first().copied().unwrap_or_default(),
        groups - 1,
        gpu_sums.last().copied().unwrap_or_default(),
        args.count - (reduce_groups - 1) * reduce_size
    );
    if cpu_sums != gpu_sums {
        return Err("reduce: the group sums differ from the CPU".into());
    }
    if cpu_total != gpu_total {
        return Err(format!("reduce: total {gpu_total}, CPU {cpu_total}").into());
    }
    println!(
        "  all {groups} group sums equal the CPU's; their total {gpu_total} = the CPU's sum of all {n} values"
    );
    // #endregion
    Ok(())
}

/// The bytes of a float slice, for the memcpy into mapped memory.
fn bytemuck_f32(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The bytes of a u32 slice, for the memcpy into mapped memory.
fn bytemuck_u32(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 25 compute")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);

    // Validation setup from step 02.
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    // SAFETY: plain query, no pointers involved.
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    if !layers.iter().any(|l| {
        l.layer_name_as_c_str()
            .is_ok_and(|name| name == validation_layer)
    }) {
        return Err("VK_LAYER_KHRONOS_validation not found. Install the Vulkan SDK.".into());
    }
    let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));
    let layer_ptrs = [validation_layer.as_ptr()];
    let ext_ptrs = [debug_utils::NAME.as_ptr()];
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut messenger_info);
    // SAFETY: `create_info` and everything it points to outlive the call.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let debug_utils = debug_utils::Instance::load(&entry, &instance);
    // SAFETY: `messenger_info` is valid and the callback is a valid `extern "system"` fn.
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&messenger_info, None)? };

    // SAFETY: `instance` is alive.
    let devices = unsafe { instance.enumerate_physical_devices()? };
    if devices.is_empty() {
        return Err("No Vulkan-capable GPU found (is the driver installed?)".into());
    }
    println!("Found {} GPU(s)", devices.len());
    for (i, &pd) in devices.iter().enumerate() {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        println!(
            "GPU {i}: {}",
            props.device_name_as_c_str()?.to_string_lossy()
        );
    }

    // Step 05's choice: a Vulkan 1.4 GPU with synchronization2 (for
    // vkCmdPipelineBarrier2, vkQueueSubmit2, vkCmdWriteTimestamp2), discrete first.
    let mut best: Option<(u32, vk::PhysicalDevice)> = None;
    for &pd in &devices {
        // SAFETY: `pd` was enumerated from this instance.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.api_version < vk::API_VERSION_1_4 {
            continue;
        }
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut f13);
        // SAFETY: `pd` came from this instance, and the chain structs outlive the call.
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        if f13.synchronization2 == 0 {
            continue;
        }
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            _ => 1,
        };
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, pd));
        }
    }
    let (_, physical_device) = best.ok_or("No GPU supports Vulkan 1.4 with synchronization2")?;
    // SAFETY: `physical_device` was enumerated from this instance.
    let chosen = unsafe { instance.get_physical_device_properties(physical_device) };
    println!(
        "\nPicked: {}",
        chosen.device_name_as_c_str()?.to_string_lossy()
    );

    // #region queue-family
    // Any family with COMPUTE can run dispatches; on desktop GPUs the
    // graphics family has COMPUTE too (family 0 on the RTX 3080 Ti). The
    // first one is enough here; step 27 looks for a family WITHOUT graphics.
    // SAFETY: `physical_device` was enumerated from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let family = u32::try_from(
        families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("No compute queue family")?,
    )?;
    println!("Compute queue family: {family} (the first with COMPUTE)");
    // #endregion
    let priorities = [1.0_f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .push(&mut features13);
    // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    println!("VkDevice created with synchronization2");

    let result = run(&instance, physical_device, &device, family, &args);

    // #region cleanup
    // `run` destroyed its objects (Objects::drop) whether it succeeded or
    // not; what remains is the device, the messenger and the instance.
    // SAFETY: nothing uses these objects any more; each is destroyed once.
    unsafe {
        device.destroy_device(None);
        debug_utils.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    // #endregion
    result
}
