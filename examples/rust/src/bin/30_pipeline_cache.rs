//! Step 30 - Pipeline cache and pipeline binaries.
//!
//! No window, no drawing: this step only CREATES pipelines and measures how
//! long that takes. vkCreateGraphicsPipelines is where the driver compiles
//! SPIR-V into GPU machine code, and a real application creates hundreds of
//! pipelines (materials x passes x vertex formats). Three ways to avoid
//! paying for the compile again:
//!
//!   VkPipelineCache        an object the driver fills with compiled code;
//!                          vkGetPipelineCacheData turns it into bytes the
//!                          application stores, and the next run hands them back
//!   creation feedback      VkPipelineCreationFeedbackCreateInfo: the driver
//!                          reports whether the cache was hit, and how long it took
//!   pipeline binaries      VK_KHR_pipeline_binary: the compiled code per pipeline,
//!                          as objects the application keys, stores and reloads
//!
//! The workload: `--count N` (default 64) variants of one graphics pipeline,
//! made different by five specialization constants of a PBR-like fragment
//! shader (shaders/pbr_variants.slang). Phases:
//!
//!   1. no VkPipelineCache
//!   2. a VkPipelineCache, empty or loaded from `--load FILE` (header checked)
//!   3. the same VkPipelineCache again; `--save FILE` writes its data
//!
//! `--probe` asks for creation WITHOUT compiling first
//! (FAIL_ON_PIPELINE_COMPILE_REQUIRED), `--no-driver-cache` turns off the
//! driver's own internal cache (VK_KHR_pipeline_binary), `--binary` captures
//! and reloads pipeline binaries, `--verbose` prints every pipeline,
//! `--no-validation` leaves the validation layer out (for timing: the layer
//! adds its own work to every create call). Lines that depend on timing start
//! with `[timing]`. Steps 01-05 are repeated, reduced to what this step needs.

use std::error::Error;
use std::ffi::{CStr, c_void};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ash::{
    Entry,
    ext::debug_utils,
    khr::pipeline_binary,
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
/// Command line, see the module comment.
struct Args {
    count: u32,
    probe: bool,
    save: Option<PathBuf>,
    load: Option<PathBuf>,
    no_driver_cache: bool,
    binary: bool,
    validation: bool,
    verbose: bool,
}

/// The largest `--count`: VARIANT is a 32-bit constant, so this is only a
/// bound on the run time (a cold compile takes milliseconds per pipeline).
const MAX_COUNT: u32 = 4096;

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let usage = "usage: 30_pipeline_cache [--count N] [--probe] [--save FILE] [--load FILE] \
                     [--no-driver-cache] [--binary] [--no-validation] [--verbose]";
        let mut args = Args {
            count: 64,
            probe: false,
            save: None,
            load: None,
            no_driver_cache: false,
            binary: false,
            validation: true,
            verbose: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--count" => args.count = it.next().ok_or(usage)?.parse()?,
                "--probe" => args.probe = true,
                "--save" => args.save = Some(it.next().ok_or(usage)?.into()),
                "--load" => args.load = Some(it.next().ok_or(usage)?.into()),
                "--no-driver-cache" => args.no_driver_cache = true,
                "--binary" => args.binary = true,
                "--no-validation" => args.validation = false,
                "--verbose" => args.verbose = true,
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

// #region spec-constants
/// The five specialization constants of pbr_variants.slang, in SpecId order:
/// LIGHTS, OCTAVES, BRDF, TONEMAP, VARIANT (all 32-bit ints in the shader).
const SPEC_NAMES: [&str; 5] = ["LIGHTS", "OCTAVES", "BRDF", "TONEMAP", "VARIANT"];

/// The constant values of pipeline `variant`. The first 64 variants are every
/// combination of 4 light counts x 4 octave counts x 2 BRDFs x 2 tone maps;
/// VARIANT = the index makes every pipeline different from every other, also
/// beyond 64, so no two of them can share a compiled result.
fn spec_values(variant: u32) -> [i32; 5] {
    let pick = [1, 2, 4, 8];
    let i = variant as usize;
    [
        pick[i % 4],
        pick[(i / 4) % 4],
        (variant / 16 % 2) as i32,
        (variant / 32 % 2) as i32,
        variant as i32,
    ]
}

/// One map entry per constant: SpecId i reads 4 bytes at offset 4 * i of the
/// data block (`spec_values` as bytes).
fn spec_entries() -> [vk::SpecializationMapEntry; 5] {
    std::array::from_fn(|i| {
        vk::SpecializationMapEntry::default()
            .constant_id(i as u32)
            .offset((i * size_of::<i32>()) as u32)
            .size(size_of::<i32>())
    })
}
// #endregion

/// How a pipeline is created.
#[derive(Clone, Copy)]
enum Mode<'b> {
    /// Compile if needed.
    Normal,
    /// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT: return
    /// VK_PIPELINE_COMPILE_REQUIRED instead of compiling.
    FailIfCompile,
    /// VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR: keep what binaries need.
    Capture,
    /// From these pipeline binaries (VkPipelineBinaryInfoKHR).
    FromBinaries(&'b [vk::PipelineBinaryKHR]),
}

/// What the driver reports for one pipeline and its two stages.
#[derive(Clone, Copy, Default)]
struct Feedback {
    pipeline: vk::PipelineCreationFeedback,
    stages: [vk::PipelineCreationFeedback; 2],
}

/// The state every variant shares: the pipeline layout and the two modules.
struct Shared {
    layout: vk::PipelineLayout,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
}

/// The format of the (never created) color attachment the pipelines render to.
const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

// #region create-info
/// Builds the VkGraphicsPipelineCreateInfo of `variant` for `mode`, hands it
/// to `f`, and returns what `f` returned plus the creation feedback the driver
/// wrote into the chain. Everything the create info points to lives on this
/// function's stack, so it stays valid for exactly the duration of `f`.
fn with_create_info<R>(
    shared: &Shared,
    variant: u32,
    mode: Mode<'_>,
    f: impl FnOnce(&vk::GraphicsPipelineCreateInfo<'_>) -> R,
) -> (R, Feedback) {
    let values = spec_values(variant);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let entries = spec_entries();
    let spec = vk::SpecializationInfo::default()
        .map_entries(&entries)
        .data(&bytes);
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shared.vertex)
            .name(c"vertexMain"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shared.fragment)
            .name(c"fragmentMain")
            .specialization_info(&spec),
    ];
    // A full-screen triangle from SV_VulkanVertexID: no vertex buffers.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [COLOR_FORMAT];
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

    // Creation feedback (core 1.3): one record for the whole pipeline, one
    // per stage (the count must equal stageCount,
    // VUID-VkGraphicsPipelineCreateInfo-pipelineStageCreationFeedbackCount-06594).
    let mut feedback = Feedback::default();
    let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
        .pipeline_creation_feedback(&mut feedback.pipeline)
        .pipeline_stage_creation_feedbacks(&mut feedback.stages);
    let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
        .flags(vk::PipelineCreateFlags2::CAPTURE_DATA_KHR);
    let binaries: &[vk::PipelineBinaryKHR] = match mode {
        Mode::FromBinaries(b) => b,
        _ => &[],
    };
    let mut binary_info = vk::PipelineBinaryInfoKHR::default().pipeline_binaries(binaries);

    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(shared.layout)
        .push(&mut rendering)
        .push(&mut feedback_info);
    info = match mode {
        Mode::Normal => info,
        Mode::FailIfCompile => {
            info.flags(vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED)
        }
        // With VkPipelineCreateFlags2CreateInfo in the chain, its 64-bit flags
        // replace VkGraphicsPipelineCreateInfo::flags (Vulkan 1.4 /
        // maintenance5); CAPTURE_DATA exists only there.
        Mode::Capture => info.push(&mut flags2),
        Mode::FromBinaries(_) => info.push(&mut binary_info),
    };
    let result = f(&info);
    (result, feedback)
}
// #endregion

/// One created pipeline (or VK_NULL_HANDLE) with its result code, the host
/// time of the create call and the driver's feedback.
struct Created {
    pipeline: vk::Pipeline,
    result: vk::Result,
    host: Duration,
    feedback: Feedback,
}

/// Every device object the program creates. Dropping it destroys them in
/// reverse order of creation, so an early return through `?` cleans up like
/// the normal path. Nothing is ever submitted to a queue, so nothing has to
/// be waited for first.
struct Objects<'a> {
    device: &'a ash::Device,
    binary_fn: Option<&'a pipeline_binary::Device>,
    modules: Vec<vk::ShaderModule>,
    layout: vk::PipelineLayout,
    caches: Vec<vk::PipelineCache>,
    pipelines: Vec<vk::Pipeline>,
    binaries: Vec<vk::PipelineBinaryKHR>,
}

impl Objects<'_> {
    /// Destroys the pipelines of a finished phase.
    fn destroy_pipelines(&mut self) {
        for p in self.pipelines.drain(..) {
            // SAFETY: created from this device, used by nothing.
            unsafe { self.device.destroy_pipeline(p, None) };
        }
    }

    /// Destroys the pipeline binaries created so far.
    fn destroy_binaries(&mut self) {
        let Some(pb) = self.binary_fn else { return };
        for b in self.binaries.drain(..) {
            // SAFETY: created from this device by vkCreatePipelineBinariesKHR;
            // a pipeline created from a binary does not need it any more.
            unsafe { (pb.fp().destroy_pipeline_binary_khr)(pb.device(), b, std::ptr::null()) };
        }
    }

    /// Destroys a pipeline cache this program created.
    fn destroy_cache(&mut self, cache: vk::PipelineCache) {
        self.caches.retain(|&c| c != cache);
        // SAFETY: created from this device; no create call uses it any more.
        unsafe { self.device.destroy_pipeline_cache(cache, None) };
    }

    /// Creates a pipeline cache, empty or from `initial` (data that passed
    /// `check_header`).
    fn create_cache(&mut self, initial: &[u8]) -> Result<vk::PipelineCache, Box<dyn Error>> {
        let info = vk::PipelineCacheCreateInfo::default().initial_data(initial);
        // SAFETY: `initial` is empty or the bytes vkGetPipelineCacheData
        // wrote, whose header was checked against this device.
        let cache = unsafe { self.device.create_pipeline_cache(&info, None)? };
        self.caches.push(cache);
        Ok(cache)
    }

    // #region create-pipeline
    /// Creates pipeline `variant` with one vkCreateGraphicsPipelines call,
    /// timed on the host. VK_PIPELINE_COMPILE_REQUIRED is a SUCCESS code
    /// (positive), returned only with FAIL_ON_PIPELINE_COMPILE_REQUIRED, and
    /// then the pipeline is VK_NULL_HANDLE; every other non-success code is
    /// an error.
    fn create_pipeline(
        &mut self,
        shared: &Shared,
        variant: u32,
        cache: vk::PipelineCache,
        mode: Mode<'_>,
    ) -> Result<Created, Box<dyn Error>> {
        let device = self.device;
        let ((created, host), feedback) = with_create_info(shared, variant, mode, |info| {
            let start = Instant::now();
            // SAFETY: all create-info data outlives the call; the cache and
            // the binaries belong to this device.
            let created = unsafe {
                device.create_graphics_pipelines(cache, std::slice::from_ref(info), None)
            };
            (created, start.elapsed())
        });
        let (pipeline, result) = match created {
            Ok(p) => (p.first().copied(), vk::Result::SUCCESS),
            Err((p, vk::Result::PIPELINE_COMPILE_REQUIRED)) => {
                (p.first().copied(), vk::Result::PIPELINE_COMPILE_REQUIRED)
            }
            Err((_, e)) => return Err(e.into()),
        };
        let pipeline = pipeline.ok_or("vkCreateGraphicsPipelines returned no pipeline")?;
        if pipeline != vk::Pipeline::null() {
            self.pipelines.push(pipeline);
        }
        Ok(Created {
            pipeline,
            result,
            host,
            feedback,
        })
    }
    // #endregion
}

impl Drop for Objects<'_> {
    fn drop(&mut self) {
        self.destroy_pipelines();
        self.destroy_binaries();
        let device = self.device;
        // SAFETY: every handle was created from this device, is destroyed
        // exactly once and is used by no pending call (nothing was submitted).
        unsafe {
            for &c in &self.caches {
                device.destroy_pipeline_cache(c, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            for &m in &self.modules {
                device.destroy_shader_module(m, None);
            }
        }
    }
}

/// `VK_` + the result's name, as the C headers spell it.
fn result_name(r: vk::Result) -> String {
    format!("VK_{r:?}")
}

/// The feedback flags of interest as text.
fn flag_names(flags: vk::PipelineCreationFeedbackFlags) -> String {
    let table = [
        (vk::PipelineCreationFeedbackFlags::VALID, "VALID"),
        (
            vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT,
            "APPLICATION_PIPELINE_CACHE_HIT",
        ),
        (
            vk::PipelineCreationFeedbackFlags::BASE_PIPELINE_ACCELERATION,
            "BASE_PIPELINE_ACCELERATION",
        ),
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

/// Median of a non-empty list of times, in milliseconds.
fn median_ms(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    match (sorted.get(mid.wrapping_sub(1)), sorted.get(mid)) {
        (Some(a), Some(b)) if sorted.len() % 2 == 0 => (a + b) / 2.0,
        (_, Some(m)) => *m,
        _ => 0.0,
    }
}

// #region phase
/// Creates all `count` variants into `cache` (VK_NULL_HANDLE = no cache) and
/// prints what the driver reported and how long it took. The pipelines stay
/// in `objects` for the caller (binary capture uses them).
fn run_phase<'b>(
    objects: &mut Objects,
    shared: &Shared,
    count: u32,
    cache: vk::PipelineCache,
    mode: &dyn Fn(u32) -> Result<Mode<'b>, Box<dyn Error>>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut created = Vec::new();
    for variant in 0..count {
        let c = objects.create_pipeline(shared, variant, cache, mode(variant)?)?;
        if c.result != vk::Result::SUCCESS {
            return Err(format!("variant {variant}: {}", result_name(c.result)).into());
        }
        created.push(c);
    }
    let hits = |f: fn(&Feedback) -> vk::PipelineCreationFeedbackFlags| {
        created
            .iter()
            .filter(|c| {
                f(&c.feedback)
                    .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT)
            })
            .count()
    };
    let valid = created
        .iter()
        .filter(|c| {
            c.feedback
                .pipeline
                .flags
                .contains(vk::PipelineCreationFeedbackFlags::VALID)
        })
        .count();
    println!(
        "  feedback: VALID {valid}/{count}, APPLICATION_PIPELINE_CACHE_HIT {}/{count} (stages: vertex {}/{count}, fragment {}/{count})",
        hits(|f| f.pipeline.flags),
        hits(|f| f.stages[0].flags),
        hits(|f| f.stages[1].flags),
    );
    let host: Vec<f64> = created.iter().map(|c| c.host.as_secs_f64() * 1e3).collect();
    let driver: Vec<f64> = created
        .iter()
        .map(|c| c.feedback.pipeline.duration as f64 / 1e6)
        .collect();
    let (min, max) = host
        .iter()
        .fold((f64::MAX, 0.0_f64), |(lo, hi), &t| (lo.min(t), hi.max(t)));
    println!(
        "[timing]   host: {:.1} ms total, per pipeline median {:.3} ms (min {min:.3}, max {max:.3}); driver-reported median {:.3} ms",
        host.iter().sum::<f64>(),
        median_ms(&host),
        median_ms(&driver),
    );
    if verbose {
        for (variant, c) in (0..count).zip(&created) {
            let v = spec_values(variant);
            let constants: Vec<String> = SPEC_NAMES
                .iter()
                .zip(v)
                .take(4)
                .map(|(n, x)| format!("{n} {x}"))
                .collect();
            println!(
                "[timing]     variant {variant} ({}): host {:.3} ms, driver {:.3} ms, {}",
                constants.join(", "),
                c.host.as_secs_f64() * 1e3,
                c.feedback.pipeline.duration as f64 / 1e6,
                flag_names(c.feedback.pipeline.flags)
            );
        }
    }
    Ok(())
}
// #endregion

// #region header-check
/// VkPipelineCacheHeaderVersionOne: the first 32 bytes of pipeline cache
/// data, every field little-endian regardless of the host ("written with the
/// least significant byte first").
struct CacheHeader {
    header_size: u32,
    header_version: u32,
    vendor_id: u32,
    device_id: u32,
    uuid: [u8; vk::UUID_SIZE],
}

/// The header of `data`, or None if it is shorter than 32 bytes.
fn parse_header(data: &[u8]) -> Option<CacheHeader> {
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
    };
    Some(CacheHeader {
        header_size: word(0)?,
        header_version: word(4)?,
        vendor_id: word(8)?,
        device_id: word(12)?,
        uuid: data.get(16..32)?.try_into().ok()?,
    })
}

/// The UUID as text, its 16 bytes in stored order, grouped 4-2-2-2-6.
fn uuid_text(uuid: &[u8; vk::UUID_SIZE]) -> String {
    let hex: Vec<String> = uuid.iter().map(|b| format!("{b:02x}")).collect();
    [0..4, 4..6, 6..8, 8..10, 10..16]
        .map(|r| hex.get(r).map(|s| s.concat()).unwrap_or_default())
        .join("-")
}

fn print_header(h: &CacheHeader) {
    println!(
        "  header: headerSize {}, headerVersion {}, vendorID 0x{:04X}, deviceID 0x{:04X}, pipelineCacheUUID {}",
        h.header_size,
        h.header_version,
        h.vendor_id,
        h.device_id,
        uuid_text(&h.uuid)
    );
}

/// Accepts cache data only if its header describes this device: the driver
/// would treat foreign data as an empty cache anyway ("If the pipeline cache
/// data is incompatible ... the pipeline cache will be initially empty"),
/// but checking here lets the program say WHY it starts empty, and keeps
/// bytes that did not come from vkGetPipelineCacheData away from the driver
/// (VUID-VkPipelineCacheCreateInfo-initialDataSize-00769).
fn check_header(data: &[u8], props: &vk::PhysicalDeviceProperties) -> Result<(), String> {
    let h = parse_header(data).ok_or(format!(
        "{} bytes, shorter than the 32-byte header",
        data.len()
    ))?;
    print_header(&h);
    if h.header_size != 32 {
        // VUID-VkPipelineCacheHeaderVersionOne-headerSize-04967: must be 32.
        return Err(format!("headerSize {} (must be 32)", h.header_size));
    }
    if h.header_version != vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32 {
        // VUID-VkPipelineCacheHeaderVersionOne-headerVersion-04968.
        return Err(format!(
            "headerVersion {} (this program reads version ONE = 1)",
            h.header_version
        ));
    }
    if h.vendor_id != props.vendor_id || h.device_id != props.device_id {
        return Err(format!(
            "written for vendorID 0x{:04X} deviceID 0x{:04X}, this GPU is 0x{:04X} 0x{:04X}",
            h.vendor_id, h.device_id, props.vendor_id, props.device_id
        ));
    }
    if h.uuid != props.pipeline_cache_uuid {
        return Err("pipelineCacheUUID differs (another driver build wrote it)".into());
    }
    Ok(())
}
// #endregion

// #region load-cache
/// The initial data of the phase-2 cache: `--load FILE` if the file exists
/// and its header matches this device, else nothing (an empty cache). A
/// missing or rejected file is the normal first-run case for an application,
/// so it is reported, not an error; a file that exists but cannot be read is.
fn load_cache(
    path: &Path,
    props: &vk::PhysicalDeviceProperties,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!(
                "Loading {}: not found; starting with an empty cache",
                path.display()
            );
            return Ok(Vec::new());
        }
        Err(e) => return Err(format!("{}: {e}", path.display()).into()),
    };
    println!("Loading {}: {} bytes", path.display(), data.len());
    match check_header(&data, props) {
        Ok(()) => {
            println!("  accepted: the header matches this device");
            Ok(data)
        }
        Err(why) => {
            println!("  rejected: {why}; starting with an empty cache");
            Ok(Vec::new())
        }
    }
}
// #endregion

// #region save-cache
/// Writes `data` to `path` through a temporary file and a rename, so a crash
/// halfway never leaves a truncated cache behind for the next run.
fn save_cache(path: &Path, data: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    println!("Saved {} bytes to {}", data.len(), path.display());
    Ok(())
}
// #endregion

// #region binary-api
/// The pipeline key (vkGetPipelineKeyKHR) of a create info, or with None the
/// GLOBAL key: if that changes between saving and loading, every saved
/// binary is invalid.
fn pipeline_key(
    pb: &pipeline_binary::Device,
    info: Option<&vk::GraphicsPipelineCreateInfo<'_>>,
) -> Result<vk::PipelineBinaryKeyKHR<'static>, vk::Result> {
    // VkPipelineCreateInfoKHR only points at the real create info (pNext is
    // declared non-const in the C struct; the driver only reads it).
    let wrapper = info.map(|i| vk::PipelineCreateInfoKHR {
        p_next: std::ptr::from_ref(i).cast_mut().cast(),
        ..Default::default()
    });
    let mut key = vk::PipelineBinaryKeyKHR::default();
    // SAFETY: `wrapper` (or null) and `key` outlive the call.
    unsafe {
        (pb.fp().get_pipeline_key_khr)(
            pb.device(),
            wrapper
                .as_ref()
                .map_or(std::ptr::null(), std::ptr::from_ref),
            &mut key,
        )
    }
    .result()?;
    Ok(key)
}

/// The bytes of a key that are valid (`keySize` of the 32).
fn key_bytes<'k>(key: &'k vk::PipelineBinaryKeyKHR<'_>) -> &'k [u8] {
    key.key.get(..key.key_size as usize).unwrap_or(&key.key)
}

/// vkCreatePipelineBinariesKHR from `info`, with the two-call idiom when the
/// count is not known (a pipeline source). Every handle created is added to
/// `objects` before the result is checked, so a partial failure still
/// destroys what was created.
fn create_binaries(
    objects: &mut Objects,
    info: &vk::PipelineBinaryCreateInfoKHR<'_>,
    known_count: Option<u32>,
) -> Result<Vec<vk::PipelineBinaryKHR>, Box<dyn Error>> {
    let pb = objects
        .binary_fn
        .ok_or("VK_KHR_pipeline_binary is not enabled")?;
    let create = |handles: &mut vk::PipelineBinaryHandlesInfoKHR<'_>| {
        // SAFETY: `info` and everything it points to outlive the call;
        // `handles` points at an array of its count (or null to query it).
        unsafe {
            (pb.fp().create_pipeline_binaries_khr)(pb.device(), info, std::ptr::null(), handles)
        }
    };
    let count = match known_count {
        Some(n) => n,
        None => {
            let mut query = vk::PipelineBinaryHandlesInfoKHR::default();
            create(&mut query).result()?;
            query.pipeline_binary_count
        }
    };
    let mut out = vec![vk::PipelineBinaryKHR::null(); count as usize];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR {
        pipeline_binary_count: count,
        p_pipeline_binaries: out.as_mut_ptr(),
        ..Default::default()
    };
    let result = create(&mut handles);
    out.truncate(handles.pipeline_binary_count as usize);
    objects.binaries.extend(
        out.iter()
            .copied()
            .filter(|&b| b != vk::PipelineBinaryKHR::null()),
    );
    if result != vk::Result::SUCCESS {
        return Err(format!("vkCreatePipelineBinariesKHR: {}", result_name(result)).into());
    }
    Ok(out)
}

/// The key and the data of one binary (vkGetPipelineBinaryDataKHR, two calls).
fn binary_data(
    pb: &pipeline_binary::Device,
    binary: vk::PipelineBinaryKHR,
) -> Result<(vk::PipelineBinaryKeyKHR<'static>, Vec<u8>), vk::Result> {
    let info = vk::PipelineBinaryDataInfoKHR::default().pipeline_binary(binary);
    let mut key = vk::PipelineBinaryKeyKHR::default();
    let mut size = 0;
    // SAFETY: all pointers outlive the calls; the second call gets a buffer
    // of exactly the size the first one returned.
    unsafe {
        (pb.fp().get_pipeline_binary_data_khr)(
            pb.device(),
            &info,
            &mut key,
            &mut size,
            std::ptr::null_mut(),
        )
        .result()?;
        let mut data = vec![0_u8; size];
        (pb.fp().get_pipeline_binary_data_khr)(
            pb.device(),
            &info,
            &mut key,
            &mut size,
            data.as_mut_ptr().cast(),
        )
        .result()?;
        data.truncate(size);
        Ok((key, data))
    }
}

/// vkReleaseCapturedPipelineDataKHR: the pipeline no longer needs to keep
/// what CAPTURE_DATA made it keep.
fn release_captured(
    pb: &pipeline_binary::Device,
    pipeline: vk::Pipeline,
) -> Result<(), vk::Result> {
    let info = vk::ReleaseCapturedPipelineDataInfoKHR::default().pipeline(pipeline);
    // SAFETY: `pipeline` was created with CAPTURE_DATA and is released once.
    unsafe { (pb.fp().release_captured_pipeline_data_khr)(pb.device(), &info, std::ptr::null()) }
        .result()
}
// #endregion

/// One captured binary: what an application would write to disk.
struct SavedBinary {
    key: vk::PipelineBinaryKeyKHR<'static>,
    data: Vec<u8>,
}

/// Everything after device creation. `objects` owns what it creates, so any
/// error returns through its Drop.
fn run(
    device: &ash::Device,
    binary_fn: Option<&pipeline_binary::Device>,
    props: &vk::PhysicalDeviceProperties,
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    let mut objects = Objects {
        device,
        binary_fn,
        modules: Vec::new(),
        layout: vk::PipelineLayout::null(),
        caches: Vec::new(),
        pipelines: Vec::new(),
        binaries: Vec::new(),
    };

    // The SPIR-V is compiled from shaders/pbr_variants.slang by `bun run
    // sync:shaders` and embedded at build time (step 08 explains `read_spv`).
    const SPV: &[u8] = include_bytes!("../../../../shaders/spv/pbr_variants.spv");
    let words = ash::util::read_spv(&mut std::io::Cursor::new(SPV))?;
    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    // One module holds both entry points; the pipelines reference it twice.
    // SAFETY: valid SPIR-V (spirv-val in sync:shaders); plain create call.
    let module = unsafe { device.create_shader_module(&module_info, None)? };
    objects.modules.push(module);
    // No descriptors, no push constants: the fragment shader needs only its
    // specialization constants and the interpolated uv.
    // SAFETY: plain create call on a live device.
    objects.layout =
        unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)? };
    let shared = Shared {
        layout: objects.layout,
        vertex: module,
        fragment: module,
    };
    println!(
        "\nShader pbr_variants.spv: {} bytes, entry points vertexMain + fragmentMain, specialization constants {}",
        SPV.len(),
        SPEC_NAMES.join(", ")
    );
    println!(
        "{} pipeline variants (dynamic rendering, color {COLOR_FORMAT:?}); the first 64 are LIGHTS 1/2/4/8 x OCTAVES 1/2/4/8 x BRDF 0/1 x TONEMAP 0/1",
        args.count
    );

    // #region probe
    // Can the pipeline be created WITHOUT compiling? With
    // FAIL_ON_PIPELINE_COMPILE_REQUIRED the driver answers instead of
    // compiling: VK_PIPELINE_COMPILE_REQUIRED and VK_NULL_HANDLE if it would
    // have to compile (an engine would then compile on a worker thread and
    // draw with a fallback meanwhile). The probe uses its own cache and the
    // variant right after the phases' range, so it warms nothing they create.
    if args.probe {
        let variant = args.count;
        println!(
            "\nProbe (variant {variant}, its own empty VkPipelineCache, FAIL_ON_PIPELINE_COMPILE_REQUIRED):"
        );
        let cache = objects.create_cache(&[])?;
        let first = objects.create_pipeline(&shared, variant, cache, Mode::FailIfCompile)?;
        let handle = |c: &Created| {
            if c.pipeline == vk::Pipeline::null() {
                "VK_NULL_HANDLE"
            } else {
                "a pipeline"
            }
        };
        println!(
            "  1. empty cache: {}, pipeline = {}",
            result_name(first.result),
            handle(&first)
        );
        if first.result == vk::Result::SUCCESS {
            println!(
                "     (nothing to compile although the application's cache is empty: the driver's internal cache had it)"
            );
        }
        let normal = objects.create_pipeline(&shared, variant, cache, Mode::Normal)?;
        println!(
            "  2. without the flag, same cache: {} (created, and now in the cache)",
            result_name(normal.result)
        );
        let again = objects.create_pipeline(&shared, variant, cache, Mode::FailIfCompile)?;
        println!(
            "  3. with the flag again: {}, pipeline = {}",
            result_name(again.result),
            handle(&again)
        );
        objects.destroy_pipelines();
        objects.destroy_cache(cache);
    }
    // #endregion

    // #region phases
    let normal = |_: u32| -> Result<Mode<'_>, Box<dyn Error>> { Ok(Mode::Normal) };
    println!("\nPhase 1: no VkPipelineCache (pipelineCache = VK_NULL_HANDLE)");
    run_phase(
        &mut objects,
        &shared,
        args.count,
        vk::PipelineCache::null(),
        &normal,
        args.verbose,
    )?;
    objects.destroy_pipelines();

    let initial = match &args.load {
        Some(path) => {
            println!();
            load_cache(path, props)?
        }
        None => Vec::new(),
    };
    let cache = objects.create_cache(&initial)?;
    if initial.is_empty() {
        println!("\nPhase 2: a new, empty VkPipelineCache");
    } else {
        println!(
            "\nPhase 2: VkPipelineCache created from the file ({} bytes)",
            initial.len()
        );
    }
    run_phase(
        &mut objects,
        &shared,
        args.count,
        cache,
        &normal,
        args.verbose,
    )?;
    objects.destroy_pipelines();

    println!("\nPhase 3: the same VkPipelineCache again");
    run_phase(
        &mut objects,
        &shared,
        args.count,
        cache,
        &normal,
        args.verbose,
    )?;
    objects.destroy_pipelines();
    // #endregion

    // #region cache-data
    // vkGetPipelineCacheData: everything the cache holds, as bytes that start
    // with the 32-byte header. ash runs the two-call idiom (size, then data).
    // SAFETY: the cache belongs to this device; no create call uses it now.
    let data = unsafe { device.get_pipeline_cache_data(cache)? };
    println!("\nVkPipelineCache data: {} bytes", data.len());
    check_header(&data, props).map_err(|why| format!("our own cache data: {why}"))?;
    if let Some(path) = &args.save {
        save_cache(path, &data)?;
    }
    objects.destroy_cache(cache);
    // #endregion

    if args.binary {
        binaries(&mut objects, &shared, args)?;
    }
    Ok(())
}

/// The `--binary` part: capture binaries from freshly created pipelines,
/// drop everything, recreate the pipelines from the saved bytes.
fn binaries(objects: &mut Objects, shared: &Shared, args: &Args) -> Result<(), Box<dyn Error>> {
    let pb = objects
        .binary_fn
        .ok_or("--binary needs VK_KHR_pipeline_binary")?;
    let count = args.count;
    println!("\nPipeline binaries (VK_KHR_pipeline_binary):");

    // #region binary-keys
    // The global key identifies the driver's binary format: an application
    // stores it next to the binaries and discards them if it changed. The
    // pipeline key identifies one create info (here: one variant), the name
    // under which its binaries are stored.
    let global = pipeline_key(pb, None)?;
    println!(
        "  global key: {} bytes (stored with the binaries; a different key on load = discard them)",
        global.key_size
    );
    let mut keys = Vec::new();
    for variant in 0..count {
        let (key, _) = with_create_info(shared, variant, Mode::Normal, |info| {
            pipeline_key(pb, Some(info))
        });
        keys.push(key_bytes(&key?).to_vec());
    }
    let mut distinct = keys.clone();
    distinct.sort();
    distinct.dedup();
    println!(
        "  pipeline keys: {count} create infos -> {} distinct keys",
        distinct.len()
    );
    // #endregion

    // #region binary-capture
    // CAPTURE_DATA makes each pipeline keep what vkCreatePipelineBinariesKHR
    // needs; such pipelines must be created without a VkPipelineCache
    // (VUID-vkCreateGraphicsPipelines-pNext-09617).
    println!("\n  Capture: {count} pipelines with CAPTURE_DATA, no VkPipelineCache");
    run_phase(
        objects,
        shared,
        count,
        vk::PipelineCache::null(),
        &|_| Ok(Mode::Capture),
        args.verbose,
    )?;
    let start = Instant::now();
    let pipelines = objects.pipelines.clone();
    let mut saved: Vec<Vec<SavedBinary>> = Vec::new();
    for &pipeline in &pipelines {
        let info = vk::PipelineBinaryCreateInfoKHR::default().pipeline(pipeline);
        let handles = create_binaries(objects, &info, None)?;
        let mut per_pipeline = Vec::new();
        for &b in &handles {
            let (key, data) = binary_data(pb, b)?;
            per_pipeline.push(SavedBinary { key, data });
        }
        release_captured(pb, pipeline)?;
        saved.push(per_pipeline);
    }
    let extract = start.elapsed();
    objects.destroy_pipelines();
    objects.destroy_binaries();
    let total: usize = saved.iter().map(Vec::len).sum();
    let bytes: usize = saved.iter().flatten().map(|b| b.data.len()).sum();
    let per: Vec<usize> = saved.iter().map(Vec::len).collect();
    let same = per.windows(2).all(|w| w.first() == w.get(1));
    println!(
        "  captured: {total} binaries ({}), {bytes} bytes of binary data",
        if same {
            format!("{} per pipeline", per.first().copied().unwrap_or(0))
        } else {
            "a different number per pipeline".to_owned()
        }
    );
    println!(
        "[timing]   vkCreatePipelineBinariesKHR + vkGetPipelineBinaryDataKHR for all: {:.1} ms",
        extract.as_secs_f64() * 1e3
    );
    // #endregion

    // #region binary-reload
    // What a later run does with the stored bytes: binaries from keys + data,
    // then pipelines from the binaries. The driver must use the binaries
    // "instead of recalculating" (spec of VkPipelineBinaryInfoKHR): no
    // compile, and no VkPipelineCache (VUID-vkCreateGraphicsPipelines-pNext-09616).
    let start = Instant::now();
    let mut per_variant: Vec<Vec<vk::PipelineBinaryKHR>> = Vec::new();
    for stored in &saved {
        let keys: Vec<vk::PipelineBinaryKeyKHR> = stored.iter().map(|b| b.key).collect();
        let data: Vec<vk::PipelineBinaryDataKHR> = stored
            .iter()
            .map(|b| vk::PipelineBinaryDataKHR {
                data_size: b.data.len(),
                // Declared `void*` in C; the driver only reads it.
                p_data: b.data.as_ptr().cast_mut().cast(),
                ..Default::default()
            })
            .collect();
        let keys_and_data = vk::PipelineBinaryKeysAndDataKHR::default()
            .pipeline_binary_keys(&keys)
            .pipeline_binary_data(&data);
        let info = vk::PipelineBinaryCreateInfoKHR::default().keys_and_data_info(&keys_and_data);
        let n = u32::try_from(stored.len())?;
        per_variant.push(create_binaries(objects, &info, Some(n))?);
    }
    println!(
        "\n  Reload: {total} binaries from the saved keys + data, then {count} pipelines from them"
    );
    println!(
        "[timing]   vkCreatePipelineBinariesKHR from data for all: {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
    run_phase(
        objects,
        shared,
        count,
        vk::PipelineCache::null(),
        &|variant| {
            let b = per_variant
                .get(variant as usize)
                .ok_or("variant out of range")?;
            Ok(Mode::FromBinaries(b))
        },
        args.verbose,
    )?;
    objects.destroy_pipelines();
    objects.destroy_binaries();
    // #endregion
    Ok(())
}

/// Instance, messenger and device. Dropping it destroys them in reverse
/// order, so `?` anywhere in `main` after the instance exists cleans up.
struct Context {
    entry: Entry,
    instance: ash::Instance,
    debug_utils: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: Option<ash::Device>,
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: nothing uses these objects any more (`run` destroyed its
        // objects before returning); each is destroyed once, children first.
        unsafe {
            if let Some(device) = self.device.take() {
                device.destroy_device(None);
            }
            if let Some((utils, messenger)) = self.debug_utils.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Whether `pd` offers the device extension `name`.
fn has_extension(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    name: &CStr,
) -> Result<bool, vk::Result> {
    // SAFETY: `pd` was enumerated from `instance`.
    let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
    Ok(exts
        .iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name)))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;

    // SAFETY: loads the system Vulkan loader DLL (see step 01); `entry` outlives all Vulkan objects.
    let entry = unsafe { Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 30 pipeline cache")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);

    // Validation setup from step 02, unless --no-validation.
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
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
    let mut layer_ptrs = Vec::new();
    let mut ext_ptrs = Vec::new();
    if args.validation {
        // SAFETY: plain query, no pointers involved.
        let layers = unsafe { entry.enumerate_instance_layer_properties()? };
        if !layers.iter().any(|l| {
            l.layer_name_as_c_str()
                .is_ok_and(|name| name == validation_layer)
        }) {
            return Err("VK_LAYER_KHRONOS_validation not found. Install the Vulkan SDK.".into());
        }
        layer_ptrs.push(validation_layer.as_ptr());
        ext_ptrs.push(debug_utils::NAME.as_ptr());
    }
    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs);
    if args.validation {
        create_info = create_info.push(&mut messenger_info);
    }
    // SAFETY: `create_info` and everything it points to outlive the call.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let mut ctx = Context {
        entry,
        instance,
        debug_utils: None,
        device: None,
    };
    if args.validation {
        let utils = debug_utils::Instance::load(&ctx.entry, &ctx.instance);
        // SAFETY: `messenger_info` is valid and the callback is a valid `extern "system"` fn.
        let messenger = unsafe { utils.create_debug_utils_messenger(&messenger_info, None)? };
        ctx.debug_utils = Some((utils, messenger));
    }
    let instance = &ctx.instance;

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

    // Step 05's choice: a Vulkan 1.4 GPU with dynamicRendering and
    // pipelineCreationCacheControl (both core 1.3 features, required by 1.3+;
    // checked anyway), discrete first.
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
        if f13.dynamic_rendering == 0 || f13.pipeline_creation_cache_control == 0 {
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
    let (_, physical_device) = best.ok_or(
        "No GPU supports Vulkan 1.4 with dynamicRendering and pipelineCreationCacheControl",
    )?;
    // SAFETY: `physical_device` was enumerated from this instance.
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    println!(
        "\nPicked: {}",
        props.device_name_as_c_str()?.to_string_lossy()
    );
    println!(
        "Validation: {}",
        if args.validation {
            "on (VK_LAYER_KHRONOS_validation)"
        } else {
            "off (--no-validation: the timings below are the driver's alone)"
        }
    );

    // #region cache-identity
    // What a pipeline cache header must match: compiled code depends on the
    // vendor, the device and the driver build; pipelineCacheUUID is the
    // driver's name for "my cache format and compiler" (it changes with
    // driver updates on NVIDIA, so an updated driver starts cold).
    println!("\nPipeline cache identity (VkPhysicalDeviceProperties):");
    println!(
        "  vendorID 0x{:04X}, deviceID 0x{:04X}, pipelineCacheUUID {}",
        props.vendor_id,
        props.device_id,
        uuid_text(&props.pipeline_cache_uuid)
    );
    // #endregion

    // #region device
    // VK_KHR_pipeline_binary is needed for --binary, and for --no-driver-cache:
    // VkDevicePipelineBinaryInternalCacheControlKHR is the only standard way
    // to switch off a driver's internal cache, which otherwise makes a "cold"
    // run warm (an implementation may keep compiled pipelines on its own,
    // in memory and on disk).
    let binary_supported = has_extension(instance, physical_device, pipeline_binary::NAME)?;
    let mut bin_props = vk::PhysicalDevicePipelineBinaryPropertiesKHR::default();
    if binary_supported {
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut bin_props);
        // SAFETY: the extension is supported, so the struct may be chained;
        // the chain outlives the call.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let yes = |b: vk::Bool32| if b != 0 { "true" } else { "false" };
        println!(
            "VK_KHR_pipeline_binary: internalCache {}, internalCacheControl {}, prefersInternalCache {}, precompiledInternalCache {}, compressedData {}",
            yes(bin_props.pipeline_binary_internal_cache),
            yes(bin_props.pipeline_binary_internal_cache_control),
            yes(bin_props.pipeline_binary_prefers_internal_cache),
            yes(bin_props.pipeline_binary_precompiled_internal_cache),
            yes(bin_props.pipeline_binary_compressed_data),
        );
    } else {
        println!("VK_KHR_pipeline_binary: not supported");
    }
    let need_binary_ext = args.binary || args.no_driver_cache;
    if need_binary_ext && !binary_supported {
        return Err("--binary and --no-driver-cache need VK_KHR_pipeline_binary".into());
    }
    if args.no_driver_cache && bin_props.pipeline_binary_internal_cache_control == 0 {
        // VUID-VkDevicePipelineBinaryInternalCacheControlKHR-disableInternalCache-09602.
        return Err("this driver does not allow disabling its internal cache".into());
    }

    // SAFETY: `physical_device` was enumerated from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    // Pipelines belong to the device, not to a queue, and nothing is
    // submitted; a device still needs one queue to be created with.
    let family = u32::try_from(
        families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or("No graphics queue family")?,
    )?;
    let priorities = [1.0_f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .pipeline_creation_cache_control(true);
    let mut binary_features =
        vk::PhysicalDevicePipelineBinaryFeaturesKHR::default().pipeline_binaries(true);
    let mut cache_control =
        vk::DevicePipelineBinaryInternalCacheControlKHR::default().disable_internal_cache(true);
    let mut device_exts = Vec::new();
    if need_binary_ext {
        device_exts.push(pipeline_binary::NAME.as_ptr());
    }
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_exts)
        .push(&mut features13);
    if args.binary {
        device_info = device_info.push(&mut binary_features);
    }
    if args.no_driver_cache {
        device_info = device_info.push(&mut cache_control);
    }
    // SAFETY: `physical_device` belongs to `instance`, and all create-info data outlives the call.
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    ctx.device = Some(device);
    let device = ctx.device.as_ref().ok_or("no device")?;
    println!(
        "VkDevice created with dynamicRendering, pipelineCreationCacheControl{}",
        if args.binary {
            ", pipelineBinaries"
        } else {
            ""
        }
    );
    println!(
        "Driver internal cache: {}",
        if args.no_driver_cache {
            "DISABLED (--no-driver-cache: VkDevicePipelineBinaryInternalCacheControlKHR)"
        } else {
            "default (whatever the driver does)"
        }
    );
    // #endregion

    let binary_fn = need_binary_ext.then(|| pipeline_binary::Device::load(&ctx.instance, device));
    run(device, binary_fn.as_ref(), &props, &args)

    // #region cleanup
    // `run` destroyed its objects (Objects::drop) whether it succeeded or
    // not; Context::drop destroys the device, the messenger and the instance.
    // #endregion
}
