//! Captures what the driver reports about compiled pipelines through VK_KHR_pipeline_executable_properties into
//! `src/data/pipeline-stats.json` (the occupancy-and-registers lesson): every executable (the driver's compiled
//! program per stage), every statistic it exposes, and the internal representations it can dump.
//!
//! Pipelines, all created with CAPTURE_STATISTICS | CAPTURE_INTERNAL_REPRESENTATIONS:
//! - the course's compute kernels: shaders/spv/saxpy.spv, reduce.spv, particles.spv (init, simulate, advance);
//! - one graphics pipeline: shaders/spv/cube.spv (vertexMain + fragmentMain, dynamic rendering, no window);
//! - the register-pressure series `tools/vk-probe/shaders/pressure.slang`: p8 ... p256 keep N floats alive per
//!   invocation, d64 indexes 64 floats at run time (local memory); each is also timed (GPU timestamps around one
//!   large dispatch, median of `RUNS`).
//!
//! Text representations are written to `tools/vk-probe/out/` (gitignored, can be large); the JSON keeps their names,
//! descriptions and sizes. Runs with the Khronos validation layer; any WARNING/ERROR it prints goes to stderr.
//!
//! `bun run sync:pipeline-stats` (or `cargo run --release --bin pipeline-stats -- <output.json>`).

use std::ffi::{CStr, c_char, c_void};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ash::{
    Device, Entry, Instance,
    ext::debug_utils,
    khr::pipeline_executable_properties as pep,
    vk::{self, TaggedStructure as _},
};
use serde::Serialize;
use vk_probe::{Res, out_path, pretty, today, write_out};

const SAXPY: &[u8] = include_bytes!("../../../../shaders/spv/saxpy.spv");
const REDUCE: &[u8] = include_bytes!("../../../../shaders/spv/reduce.spv");
const PARTICLES: &[u8] = include_bytes!("../../../../shaders/spv/particles.spv");
const CUBE: &[u8] = include_bytes!("../../../../shaders/spv/cube.spv");
const PRESSURE: &[u8] = include_bytes!("../../shaders/pressure.spv");

/// Register-pressure entry points of pressure.slang and the floats each keeps alive per invocation. d64 is the
/// control: 64 floats indexed at run time, so they live in local memory whatever the register budget.
const PRESSURE_KERNELS: [(&CStr, u32); 8] = [
    (c"p8", 8),
    (c"p16", 16),
    (c"p32", 32),
    (c"p64", 64),
    (c"p128", 128),
    (c"p192", 192),
    (c"p256", 256),
    (c"d64", 64),
];
/// Must match pressure.slang.
const GROUP_SIZE: u32 = 256;
const ROUNDS: u32 = 8;
/// Invocations per timed dispatch: 4 Mi = 16384 workgroups, ~200 per SM on 80 SMs, so the launch overhead is noise.
const THREADS: u32 = 1 << 22;
/// Timed runs per kernel (the median is stored); one more untimed warm-up run goes first.
const RUNS: usize = 5;

const SOURCE: &str = "Real capture on the course dev machine by tools/vk-probe (binary pipeline-stats, \
    `bun run sync:pipeline-stats`; ash at the course pin): a Vulkan 1.4 device on GPU 0 with \
    VK_KHR_pipeline_executable_properties (pipelineExecutableInfo) and the Khronos validation layer; every pipeline is \
    created with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR \
    and read back with vkGetPipelineExecutablePropertiesKHR, vkGetPipelineExecutableStatisticsKHR and \
    vkGetPipelineExecutableInternalRepresentationsKHR. Statistic names, descriptions and values are verbatim from the \
    driver. `pressure`: GPU time (vkCmdWriteTimestamp2 at ALL_COMMANDS before and after) of one dispatch of `threads` \
    invocations per kernel; runs alternate between kernels after one warm-up round; measured, not a benchmark.";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct File {
    source: &'static str,
    device: String,
    driver: String,
    date: String,
    pipelines: Vec<PipelineOut>,
    pressure: Pressure,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineOut {
    name: String,
    shader: &'static str,
    entry_points: Vec<String>,
    executables: Vec<Executable>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Executable {
    name: String,
    description: String,
    stages: String,
    subgroup_size: u32,
    statistics: Vec<Statistic>,
    internal_representations: Vec<Ir>,
}

#[derive(Serialize)]
struct Statistic {
    name: String,
    description: String,
    format: &'static str,
    value: Value,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Value {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Ir {
    name: String,
    description: String,
    is_text: bool,
    bytes: usize,
    /// Where the dump was saved, relative to the repo root.
    file: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Pressure {
    group_size: u32,
    rounds: u32,
    threads: u32,
    runs: usize,
    kernels: Vec<Timing>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Timing {
    entry: String,
    floats: u32,
    median_ms: f64,
    runs_ms: Vec<f64>,
}

/// One pipeline to build: compute (one entry point) or the cube's graphics pipeline.
enum Kind {
    Compute { entry: &'static CStr },
    Graphics,
}

struct Spec {
    name: String,
    shader: &'static str,
    spv: &'static [u8],
    kind: Kind,
    /// Storage buffers in set 0 (compute) and push constant bytes; the graphics pipeline has its own layout.
    buffers: u32,
    push: u32,
}

impl Spec {
    fn compute(
        name: &str,
        shader: &'static str,
        spv: &'static [u8],
        entry: &'static CStr,
        buffers: u32,
        push: u32,
    ) -> Self {
        Self {
            name: name.to_owned(),
            shader,
            spv,
            kind: Kind::Compute { entry },
            buffers,
            push,
        }
    }
}

/// Prints validation WARNING/ERROR messages to stderr and counts them in the `Mutex<u32>` passed as user data.
/// Never panics (called through the C ABI) and returns FALSE, as the spec requires of applications.
unsafe extern "system" fn report(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: `user` is the `Mutex<u32>` of `main`, alive until the messenger is destroyed; `data` is valid during the
    // call (checked for null) and its message is a NUL-terminated string.
    let (count, data) = unsafe { (&*user.cast::<Mutex<u32>>(), data.as_ref()) };
    if let Some(d) = data
        && !d.p_message.is_null()
    {
        let msg = unsafe { CStr::from_ptr(d.p_message) }.to_string_lossy();
        eprintln!("[validation {severity:?}] {msg}");
    }
    if let Ok(mut n) = count.lock() {
        *n += 1;
    }
    vk::FALSE
}

/// A fixed-size `char[VK_MAX_DESCRIPTION_SIZE]` field as a string (the driver pads some names with spaces).
fn text(chars: &[c_char]) -> Res<String> {
    // SAFETY: c_char and u8 have the same size; the array is a fixed-size field read as bytes.
    let bytes = unsafe { std::slice::from_raw_parts(chars.as_ptr().cast::<u8>(), chars.len()) };
    Ok(CStr::from_bytes_until_nul(bytes)?
        .to_string_lossy()
        .trim()
        .to_owned())
}

fn module(device: &Device, spv: &[u8]) -> Res<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut Cursor::new(spv))?;
    // SAFETY: valid SPIR-V words (validated by spirv-val when compiled).
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
    })
}

const CAPTURE: vk::PipelineCreateFlags = vk::PipelineCreateFlags::from_raw(
    vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR.as_raw()
        | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR.as_raw(),
);

/// Objects a pipeline owns, destroyed together at the end.
struct Built {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

impl Built {
    /// SAFETY: the objects are not in use by the GPU.
    unsafe fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Set 0 of `n` storage buffers (compute) or one uniform buffer (graphics), plus `push` bytes of push constants.
fn layout(
    device: &Device,
    desc: vk::DescriptorType,
    n: u32,
    stages: vk::ShaderStageFlags,
    push: u32,
) -> Res<(vk::DescriptorSetLayout, vk::PipelineLayout)> {
    let bindings: Vec<_> = (0..n)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(desc)
                .descriptor_count(1)
                .stage_flags(stages)
        })
        .collect();
    // SAFETY: valid create infos; the bindings outlive the calls.
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?
    };
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(stages)
        .size(push)];
    let sets = [set_layout];
    let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&sets);
    if push > 0 {
        info = info.push_constant_ranges(&ranges);
    }
    // SAFETY: as above.
    let layout = unsafe { device.create_pipeline_layout(&info, None) };
    match layout {
        Ok(l) => Ok((set_layout, l)),
        Err(e) => {
            // SAFETY: unused, created above.
            unsafe { device.destroy_descriptor_set_layout(set_layout, None) };
            Err(e.into())
        }
    }
}

fn build(device: &Device, spec: &Spec) -> Res<Built> {
    let shader = module(device, spec.spv)?;
    let built = match spec.kind {
        Kind::Compute { entry } => build_compute(device, shader, entry, spec.buffers, spec.push),
        Kind::Graphics => build_cube(device, shader),
    };
    // SAFETY: a pipeline does not need its shader modules after creation.
    unsafe { device.destroy_shader_module(shader, None) };
    built
}

fn build_compute(
    device: &Device,
    shader: vk::ShaderModule,
    entry: &CStr,
    buffers: u32,
    push: u32,
) -> Res<Built> {
    let (set_layout, layout) = layout(
        device,
        vk::DescriptorType::STORAGE_BUFFER,
        buffers,
        vk::ShaderStageFlags::COMPUTE,
        push,
    )?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .flags(CAPTURE)
        .stage(stage)
        .layout(layout);
    // SAFETY: valid create info; module, entry name and layout outlive the call.
    let pipeline =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| e)
            .map(|p| p.first().copied());
    finish(device, pipeline, layout, set_layout)
}

/// The cube of steps 22/24: vertex input stride 24 (two float3), uniform buffer at set 0 binding 0, depth test, one
/// sRGB color attachment; dynamic rendering, so no render pass and no window are needed to compile it.
fn build_cube(device: &Device, shader: vk::ShaderModule) -> Res<Built> {
    let (set_layout, layout) = layout(
        device,
        vk::DescriptorType::UNIFORM_BUFFER,
        1,
        vk::ShaderStageFlags::VERTEX,
        0,
    )?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"vertexMain"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader)
            .name(c"fragmentMain"),
    ];
    let bindings = [vk::VertexInputBindingDescription::default().stride(24)];
    let attributes = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT),
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let formats = [vk::Format::B8G8R8A8_SRGB];
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&formats)
        .depth_attachment_format(vk::Format::D32_SFLOAT);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .flags(CAPTURE)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push(&mut rendering);
    // SAFETY: valid create info; everything it points to outlives the call.
    let pipeline =
        unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
            .map_err(|(_, e)| e)
            .map(|p| p.first().copied());
    finish(device, pipeline, layout, set_layout)
}

/// Pairs a created pipeline with its layouts, or destroys the layouts if creation failed.
fn finish(
    device: &Device,
    pipeline: Result<Option<vk::Pipeline>, vk::Result>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
) -> Res<Built> {
    match pipeline {
        Ok(Some(pipeline)) => Ok(Built {
            pipeline,
            layout,
            set_layout,
        }),
        other => {
            // SAFETY: unused, created by the caller.
            unsafe {
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set_layout, None);
            }
            Err(match other {
                Err(e) => e.into(),
                _ => "pipeline creation returned no pipeline".into(),
            })
        }
    }
}

fn format_name(f: vk::PipelineExecutableStatisticFormatKHR) -> Res<&'static str> {
    Ok(match f {
        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => "BOOL32",
        vk::PipelineExecutableStatisticFormatKHR::INT64 => "INT64",
        vk::PipelineExecutableStatisticFormatKHR::UINT64 => "UINT64",
        vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => "FLOAT64",
        _ => return Err(format!("unknown statistic format {f:?}").into()),
    })
}

fn value(s: &vk::PipelineExecutableStatisticKHR<'_>) -> Res<Value> {
    // SAFETY: the union member read is the one `format` names (spec of VkPipelineExecutableStatisticValueKHR).
    Ok(unsafe {
        match s.format {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                Value::Bool(s.value.b32 != vk::FALSE)
            }
            vk::PipelineExecutableStatisticFormatKHR::INT64 => Value::Int(s.value.i64),
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => Value::Uint(s.value.u64),
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => Value::Float(s.value.f64),
            f => return Err(format!("unknown statistic format {f:?}").into()),
        }
    })
}

/// A file-name-safe version of a driver-provided name.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Everything the driver reports about one pipeline; text representations are saved under `dump_dir`.
fn inspect(
    ext: &pep::Device,
    pipeline: vk::Pipeline,
    pipe_name: &str,
    dump_dir: &Path,
) -> Res<Vec<Executable>> {
    let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
    // SAFETY: the pipeline was created with the capture flags on a device with pipelineExecutableInfo enabled.
    let n = unsafe { ext.get_pipeline_executable_properties_len(&info)? };
    let mut props = vec![vk::PipelineExecutablePropertiesKHR::default(); n];
    unsafe { ext.get_pipeline_executable_properties(&info, &mut props)? };

    let mut out = Vec::new();
    for (i, p) in props.iter().enumerate() {
        let index = u32::try_from(i)?;
        let exe = vk::PipelineExecutableInfoKHR::default()
            .pipeline(pipeline)
            .executable_index(index);

        // SAFETY: valid executable index (< n).
        let n = unsafe { ext.get_pipeline_executable_statistics_len(&exe)? };
        let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); n];
        unsafe { ext.get_pipeline_executable_statistics(&exe, &mut stats)? };
        let statistics = stats
            .iter()
            .map(|s| -> Res<Statistic> {
                Ok(Statistic {
                    name: text(&s.name)?,
                    description: text(&s.description)?,
                    format: format_name(s.format)?,
                    value: value(s)?,
                })
            })
            .collect::<Res<Vec<_>>>()?;

        // Two calls: the first with null pData fills each dataSize, the second copies the data.
        // SAFETY: valid executable index; pData is null in the first call and points to `data[k]` (of the size the
        // driver reported) in the second.
        let n = unsafe { ext.get_pipeline_executable_internal_representations_len(&exe)? };
        let mut irs = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); n];
        unsafe { ext.get_pipeline_executable_internal_representations(&exe, &mut irs)? };
        let mut data: Vec<Vec<u8>> = irs.iter().map(|r| vec![0u8; r.data_size]).collect();
        for (r, d) in irs.iter_mut().zip(data.iter_mut()) {
            r.p_data = d.as_mut_ptr().cast();
        }
        unsafe { ext.get_pipeline_executable_internal_representations(&exe, &mut irs)? };

        let exe_name = text(&p.name)?;
        let mut internal_representations = Vec::new();
        for (r, d) in irs.iter().zip(&data) {
            let name = text(&r.name)?;
            let is_text = r.is_text != vk::FALSE;
            let used = &d[..r.data_size.min(d.len())];
            let file = dump_dir.join(format!(
                "{}--{i}-{}--{}.{}",
                slug(pipe_name),
                slug(&exe_name),
                slug(&name),
                if is_text { "txt" } else { "bin" }
            ));
            // A text representation is NUL-terminated (spec); the file keeps the text only.
            let body = if is_text {
                used.split(|&b| b == 0).next().unwrap_or_default()
            } else {
                used
            };
            std::fs::write(&file, body)?;
            internal_representations.push(Ir {
                name,
                description: text(&r.description)?,
                is_text,
                bytes: r.data_size,
                file: format!(
                    "tools/vk-probe/out/{}",
                    file.file_name()
                        .map(|f| f.to_string_lossy())
                        .unwrap_or_default()
                ),
            });
        }

        out.push(Executable {
            name: exe_name,
            description: text(&p.description)?,
            stages: format!("{:?}", p.stages),
            subgroup_size: p.subgroup_size,
            statistics,
            internal_representations,
        });
    }
    Ok(out)
}

/// Device-local storage buffer for the timing runs (contents written on the GPU).
struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

fn device_buffer(
    instance: &Instance,
    gpu: vk::PhysicalDevice,
    device: &Device,
    size: u64,
) -> Res<Buffer> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
    // SAFETY: valid create info.
    let buffer = unsafe { device.create_buffer(&info, None)? };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem = unsafe { instance.get_physical_device_memory_properties(gpu) };
    let ty = mem
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .position(|(i, t)| {
            req.memory_type_bits & (1 << i) != 0
                && t.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or("no DEVICE_LOCAL memory type for a storage buffer")?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(u32::try_from(ty)?);
    // SAFETY: valid allocate info; the buffer is bound once, at offset 0 (aligned).
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(e.into());
        }
    };
    let buf = Buffer { buffer, memory };
    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe { buf.destroy(device) };
        return Err(e.into());
    }
    Ok(buf)
}

impl Buffer {
    /// SAFETY: not in use by the GPU.
    unsafe fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Everything the timing runs record into.
struct Timer<'a> {
    device: &'a Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    /// Nanoseconds per timestamp tick (VkPhysicalDeviceLimits::timestampPeriod).
    period: f64,
}

impl Timer<'_> {
    /// Records `record` into the command buffer, submits it, waits, and returns the GPU time between the two
    /// timestamps `record` is wrapped in, in milliseconds.
    fn run(&self, record: impl FnOnce(vk::CommandBuffer)) -> Res<f64> {
        let d = self.device;
        // SAFETY: the command buffer is not pending (the previous run waited for its fence); every handle is valid.
        unsafe {
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            d.cmd_reset_query_pool(self.cmd, self.queries, 0, 2);
            d.cmd_write_timestamp2(
                self.cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.queries,
                0,
            );
            record(self.cmd);
            d.cmd_write_timestamp2(
                self.cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.queries,
                1,
            );
            d.end_command_buffer(self.cmd)?;
            let cmds = [self.cmd];
            d.reset_fences(&[self.fence])?;
            d.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cmds)],
                self.fence,
            )?;
            d.wait_for_fences(&[self.fence], true, 10_000_000_000)?;
            let mut ts = [0u64; 2];
            d.get_query_pool_results(
                self.queries,
                0,
                &mut ts,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            let ticks = ts[1]
                .checked_sub(ts[0])
                .ok_or("timestamps went backwards")?;
            Ok(ticks as f64 * self.period / 1e6)
        }
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s.get(s.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Times every pressure pipeline: one dispatch of `THREADS` invocations reading `seeds` and writing `result`.
fn time_pressure(
    timer: &Timer,
    pipes: &[(&Built, u32, String)],
    set: vk::DescriptorSet,
    seeds: &Buffer,
) -> Res<Vec<Timing>> {
    let d = timer.device;
    // Seeds = 0.5 (bit pattern of the float), then make the fill visible to the compute shaders.
    timer.run(|cmd| unsafe {
        d.cmd_fill_buffer(cmd, seeds.buffer, 0, vk::WHOLE_SIZE, 0x3f00_0000);
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
        d.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barrier),
        );
    })?;
    let mut runs: Vec<Vec<f64>> = vec![Vec::new(); pipes.len()];
    for round in 0..=RUNS {
        for (k, (built, _, _)) in pipes.iter().enumerate() {
            // SAFETY (closure): the pipeline, layout and set are valid; the set's buffers are bound to memory.
            let ms = timer.run(|cmd| unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, built.pipeline);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    built.layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_push_constants(
                    cmd,
                    built.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &THREADS.to_ne_bytes(),
                );
                d.cmd_dispatch(cmd, THREADS / GROUP_SIZE, 1, 1);
            })?;
            if round > 0 {
                runs.get_mut(k).ok_or("kernel index")?.push(ms);
            }
        }
    }
    Ok(pipes
        .iter()
        .zip(runs)
        .map(|((_, floats, entry), runs_ms)| Timing {
            entry: entry.clone(),
            floats: *floats,
            median_ms: median(&runs_ms),
            runs_ms,
        })
        .collect())
}

/// Compact console table: one line per executable with every statistic as `name=value`.
fn print(pipelines: &[PipelineOut]) {
    for p in pipelines {
        for e in &p.executables {
            let stats: Vec<String> = e
                .statistics
                .iter()
                .map(|s| {
                    let v = match s.value {
                        Value::Bool(b) => b.to_string(),
                        Value::Int(i) => i.to_string(),
                        Value::Uint(u) => u.to_string(),
                        Value::Float(f) => format!("{f}"),
                    };
                    format!("{}={v}", s.name)
                })
                .collect();
            let irs: Vec<String> = e
                .internal_representations
                .iter()
                .map(|r| format!("{} ({} B)", r.name, r.bytes))
                .collect();
            println!(
                "{:<20} {:<28} {:<10} sg={:<3} {} | IR: {}",
                p.name,
                e.name,
                e.stages,
                e.subgroup_size,
                stats.join(", "),
                if irs.is_empty() {
                    "-".to_owned()
                } else {
                    irs.join(", ")
                }
            );
        }
    }
}

fn main() -> Res<()> {
    let out = out_path()?;
    let dump_dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("out");
    std::fs::create_dir_all(&dump_dir)?;
    // SAFETY: loads the system Vulkan loader; `entry` outlives every Vulkan object below.
    let entry = unsafe { Entry::load()? };

    let messages = Mutex::new(0u32);
    let app = vk::ApplicationInfo::default()
        .application_name(c"vk-probe pipeline-stats")
        .api_version(vk::API_VERSION_1_4);
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
        .pfn_user_callback(Some(report))
        .user_data((&raw const messages).cast_mut().cast());
    let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let exts = [debug_utils::NAME.as_ptr()];
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&exts)
        .push(&mut messenger_info);
    // SAFETY: the create info, everything it points to and `messages` outlive the instance.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let du = debug_utils::Instance::load(&entry, &instance);
    // SAFETY: valid create info; the callback and its user data outlive the messenger.
    let messenger = unsafe { du.create_debug_utils_messenger(&messenger_info, None)? };

    // SAFETY: plain queries on a live instance.
    let gpu = *unsafe { instance.enumerate_physical_devices()? }
        .first()
        .ok_or("no Vulkan device")?;
    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut driver_props);
    unsafe { instance.get_physical_device_properties2(gpu, &mut props2) };
    let props = props2.properties;
    let device_name = props.device_name_as_c_str()?.to_string_lossy().into_owned();
    let driver = format!(
        "{} {}",
        driver_props.driver_name_as_c_str()?.to_string_lossy(),
        driver_props.driver_info_as_c_str()?.to_string_lossy()
    );

    let has_ext = unsafe { instance.enumerate_device_extension_properties(gpu)? }
        .iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == pep::NAME));
    let mut pep_features = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push(&mut pep_features);
    unsafe { instance.get_physical_device_features2(gpu, &mut f2) };
    if !has_ext || pep_features.pipeline_executable_info == vk::FALSE {
        return Err(format!("{device_name} does not support VK_KHR_pipeline_executable_properties / pipelineExecutableInfo").into());
    }

    let families = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
    let family = families
        .iter()
        .position(|f| {
            f.queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                && f.timestamp_valid_bits > 0
        })
        .ok_or("no graphics+compute queue family with timestamps")?;
    let family = u32::try_from(family)?;

    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let dev_exts = [pep::NAME.as_ptr()];
    let mut want_pep = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
        .pipeline_executable_info(true);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let dev_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&dev_exts)
        .push(&mut want_pep)
        .push(&mut f13);
    // SAFETY: valid create info; extension and features checked above (Vulkan 1.3 features are core in 1.4).
    let device = unsafe { instance.create_device(gpu, &dev_info, None)? };
    let ext = pep::Device::load(&instance, &device);
    let queue = unsafe { device.get_device_queue(family, 0) };

    let mut specs = vec![
        Spec::compute("saxpy", "shaders/spv/saxpy.spv", SAXPY, c"saxpy", 3, 8),
        Spec::compute("reduce", "shaders/spv/reduce.spv", REDUCE, c"reduce", 2, 4),
        Spec::compute(
            "particles init",
            "shaders/spv/particles.spv",
            PARTICLES,
            c"init",
            2,
            16,
        ),
        Spec::compute(
            "particles simulate",
            "shaders/spv/particles.spv",
            PARTICLES,
            c"simulate",
            2,
            16,
        ),
        Spec::compute(
            "particles advance",
            "shaders/spv/particles.spv",
            PARTICLES,
            c"advance",
            2,
            16,
        ),
        Spec {
            name: "cube".to_owned(),
            shader: "shaders/spv/cube.spv",
            spv: CUBE,
            kind: Kind::Graphics,
            buffers: 0,
            push: 0,
        },
    ];
    let first_pressure = specs.len();
    for (entry, _) in PRESSURE_KERNELS {
        let name = format!("pressure {}", entry.to_string_lossy());
        specs.push(Spec::compute(
            &name,
            "tools/vk-probe/shaders/pressure.spv",
            PRESSURE,
            entry,
            2,
            4,
        ));
    }

    let mut built = Vec::new();
    let mut pipelines = Vec::new();
    for spec in &specs {
        let b = build(&device, spec)?;
        let executables = inspect(&ext, b.pipeline, &spec.name, &dump_dir)?;
        let entry_points = match spec.kind {
            Kind::Compute { entry } => vec![entry.to_string_lossy().into_owned()],
            Kind::Graphics => vec!["vertexMain".to_owned(), "fragmentMain".to_owned()],
        };
        pipelines.push(PipelineOut {
            name: spec.name.clone(),
            shader: spec.shader,
            entry_points,
            executables,
        });
        built.push(b);
    }
    print(&pipelines);

    // Timing of the pressure series.
    let bytes = u64::from(THREADS) * 4;
    let seeds = device_buffer(&instance, gpu, &device, bytes)?;
    let result = device_buffer(&instance, gpu, &device, bytes)?;
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(2)];
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?
    };
    // Every pressure pipeline has the same set layout shape; any of them allocates a compatible set.
    let pressure_built = built.get(first_pressure..).ok_or("pressure pipelines")?;
    let set_layout = [pressure_built
        .first()
        .ok_or("pressure pipelines")?
        .set_layout];
    let set = *unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layout),
        )?
    }
    .first()
    .ok_or("no descriptor set")?;
    let infos = [
        [vk::DescriptorBufferInfo::default()
            .buffer(seeds.buffer)
            .range(vk::WHOLE_SIZE)],
        [vk::DescriptorBufferInfo::default()
            .buffer(result.buffer)
            .range(vk::WHOLE_SIZE)],
    ];
    let writes: Vec<_> = infos
        .iter()
        .zip(0u32..)
        .map(|(info, binding)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let cmd_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let cmd = *unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .command_buffer_count(1),
        )?
    }
    .first()
    .ok_or("no command buffer")?;
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let queries = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )?
    };
    let timer = Timer {
        device: &device,
        queue,
        cmd,
        fence,
        queries,
        period: f64::from(props.limits.timestamp_period),
    };
    let pipes: Vec<(&Built, u32, String)> = pressure_built
        .iter()
        .zip(PRESSURE_KERNELS)
        .map(|(b, (entry, floats))| (b, floats, entry.to_string_lossy().into_owned()))
        .collect();
    let kernels = time_pressure(&timer, &pipes, set, &seeds)?;
    println!(
        "\nregister pressure: {THREADS} invocations, groups of {GROUP_SIZE}, {ROUNDS} rounds, median of {RUNS}"
    );
    for k in &kernels {
        let runs: Vec<String> = k.runs_ms.iter().map(|m| format!("{m:.3}")).collect();
        println!(
            "  {:<5} {:>3} floats  {:>8.3} ms  [{}]",
            k.entry,
            k.floats,
            k.median_ms,
            runs.join(" ")
        );
    }

    // SAFETY: the device is idle after the waits above; every object is destroyed once, children before parents.
    unsafe {
        device.device_wait_idle()?;
        device.destroy_query_pool(queries, None);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(cmd_pool, None);
        device.destroy_descriptor_pool(pool, None);
        seeds.destroy(&device);
        result.destroy(&device);
        for b in &built {
            b.destroy(&device);
        }
        device.destroy_device(None);
        du.destroy_debug_utils_messenger(messenger, None);
        instance.destroy_instance(None);
    }
    let warnings = messages.into_inner().map_err(|e| e.to_string())?;
    eprintln!("validation: {warnings} WARNING/ERROR message(s)");

    let file = File {
        source: SOURCE,
        device: device_name,
        driver,
        date: today()?,
        pipelines,
        pressure: Pressure {
            group_size: GROUP_SIZE,
            rounds: ROUNDS,
            threads: THREADS,
            runs: RUNS,
            kernels,
        },
    };
    write_out(&out, &format!("{}\n", pretty(&file, 0)?))
}
