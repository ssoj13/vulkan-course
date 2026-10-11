//! Captures what the loader and the validation layer report during examples/rust step 02 into
//! `src/data/validation-log.json` (the messenger-filter lab): the instance of step 02 with a messenger that asks for
//! every severity, run three times:
//! - `base`: exactly step 02 (types GENERAL | VALIDATION | PERFORMANCE, messenger destroyed before the instance);
//! - `leak`: the validation lesson's experiment, vkDestroyInstance with the messenger still alive;
//! - `dab`:  the messenger also asks for DEVICE_ADDRESS_BINDING_EXT, a type of a device extension.
//!
//! The file keeps the totals of the base run per severity and type, how many messages the two experiments add, and a
//! sample of exact messages picked by `PICKS`.
//!
//! `bun run sync:validation-log` (or `cargo run --release --bin validation-log -- <output.json>`).

use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;

use ash::{
    Entry,
    ext::debug_utils,
    vk::{self, TaggedStructure as _},
};
use serde::Serialize;
use vk_probe::{Res, out_path, pretty, today, write_out};

type Sev = vk::DebugUtilsMessageSeverityFlagsEXT;
type Ty = vk::DebugUtilsMessageTypeFlagsEXT;

/// The situation a run models; `Base` messages carry no `when` in the file.
#[derive(Clone, Copy, PartialEq)]
enum Run {
    Base,
    Leak,
    Dab,
}

impl Run {
    fn when(self) -> Option<&'static str> {
        match self {
            Run::Base => None,
            Run::Leak => Some("leak"),
            Run::Dab => Some("dab"),
        }
    }
}

/// Which messages the sample keeps, in order: the first message of the run whose text contains every needle.
/// They show each part of the story: loader settings, layer and ICD discovery, the old implicit layer's warning,
/// layer loading, the validation layer's status, its errors, and unloading. Leak errors are added after `LEAK_AT`.
const PICKS: [(Run, &[&str]); 13] = [
    (Run::Base, &["No valid vk_loader_settings"]),
    (
        Run::Base,
        &["Checking for Layer Manifest files in Registry at HKEY_LOCAL_MACHINE"],
    ),
    (Run::Base, &["bdcamvk64.json\" from registry"]),
    (
        Run::Base,
        &["VkLayer_khronos_validation.json (file version"],
    ),
    (Run::Base, &["Found ICD manifest file"]),
    (Run::Base, &["Searching for ICD drivers"]),
    (Run::Base, &["Layer VK_LAYER_bandicam_helper uses"]),
    (Run::Base, &["Loading layer library C:"]),
    (
        Run::Base,
        &["Insert instance layer \"VK_LAYER_KHRONOS_validation\""],
    ),
    (Run::Base, &["vkCreateInstance layer callstack setup to:"]),
    (Run::Dab, &["messageType has"]),
    (Run::Base, &["Khronos Validation Layer Active"]),
    (Run::Base, &["Unloading layer library", "bdcamvk64.dll"]),
];
/// The leak run's errors go before the last pick (unloading happens at the very end of a run).
const LEAK_AT: usize = PICKS.len() - 1;

/// A message as the callback received it.
struct Raw {
    severity: Sev,
    kind: Ty,
    id: String,
    message: String,
}

#[derive(Serialize)]
struct Message {
    severity: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    id: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    when: Option<&'static str>,
}

#[derive(Serialize)]
struct Count {
    severity: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    n: usize,
}

#[derive(Serialize)]
struct Extra {
    leak: usize,
    dab: usize,
}

#[derive(Serialize)]
struct Totals {
    total: usize,
    counts: Vec<Count>,
    extra: Extra,
}

#[derive(Serialize)]
struct LogFile {
    source: String,
    run: Totals,
    messages: Vec<Message>,
}

/// Collects every message into the `Mutex<Vec<Raw>>` passed as user data. Never panics (it is called through the C
/// ABI) and always returns FALSE, as the spec requires of applications.
unsafe extern "system" fn collect(
    severity: Sev,
    kind: Ty,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    let text = |p: *const c_char| {
        if p.is_null() {
            String::new()
        } else {
            // SAFETY: non-null strings of the callback data are NUL-terminated and valid during the call.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    // SAFETY: `user` is the `Mutex<Vec<Raw>>` of `run`, alive until its instance is destroyed; `data` is valid
    // during the call (checked for null).
    let (log, data) = unsafe { (&*user.cast::<Mutex<Vec<Raw>>>(), data.as_ref()) };
    if let (Some(d), Ok(mut log)) = (data, log.lock()) {
        log.push(Raw {
            severity,
            kind,
            id: text(d.p_message_id_name),
            message: text(d.p_message),
        });
    }
    vk::FALSE
}

/// Step 02's instance and messenger with every severity and the run's types; returns everything the callback got.
fn run(entry: &Entry, which: Run) -> Res<Vec<Raw>> {
    let log = Mutex::new(Vec::new());
    let mut types = Ty::GENERAL | Ty::VALIDATION | Ty::PERFORMANCE;
    if which == Run::Dab {
        types |= Ty::DEVICE_ADDRESS_BINDING;
    }
    let app = vk::ApplicationInfo::default()
        .application_name(c"Vulkan from Zero - 02 validation")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"No Engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_4);
    let mut messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(Sev::VERBOSE | Sev::INFO | Sev::WARNING | Sev::ERROR)
        .message_type(types)
        .pfn_user_callback(Some(collect))
        .user_data((&raw const log).cast_mut().cast());
    let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
    let exts = [debug_utils::NAME.as_ptr()];
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&exts)
        .push(&mut messenger_info);
    // SAFETY: the create info, everything it points to and `log` outlive the instance.
    let instance = unsafe { entry.create_instance(&create_info, None)? };
    let du = debug_utils::Instance::load(entry, &instance);
    // SAFETY: valid create info; the callback and its user data outlive the messenger.
    let messenger = unsafe { du.create_debug_utils_messenger(&messenger_info, None) };
    // SAFETY: the instance is alive and destroyed once; the leak run skips the messenger on purpose (the experiment).
    unsafe {
        if let (Ok(m), false) = (&messenger, which == Run::Leak) {
            du.destroy_debug_utils_messenger(*m, None);
        }
        instance.destroy_instance(None);
    }
    messenger?;
    Ok(log.into_inner().map_err(|e| e.to_string())?)
}

fn severity_name(s: Sev) -> Res<&'static str> {
    Ok(match s {
        Sev::VERBOSE => "VERBOSE",
        Sev::INFO => "INFO",
        Sev::WARNING => "WARNING",
        Sev::ERROR => "ERROR",
        _ => return Err(format!("unexpected severity {s:?}").into()),
    })
}

fn type_name(t: Ty) -> Res<&'static str> {
    Ok(match t {
        Ty::GENERAL => "GENERAL",
        Ty::VALIDATION => "VALIDATION",
        Ty::PERFORMANCE => "PERFORMANCE",
        Ty::DEVICE_ADDRESS_BINDING => "DEVICE_ADDRESS_BINDING_EXT",
        _ => return Err(format!("unexpected message type {t:?}").into()),
    })
}

/// A sample message as stored. The layer ends multi-line messages with a line break; the lab lays out lines itself,
/// so trailing breaks are dropped.
fn message(r: &Raw, which: Run) -> Res<Message> {
    Ok(Message {
        severity: severity_name(r.severity)?,
        kind: type_name(r.kind)?,
        id: r.id.clone(),
        message: r.message.trim_end_matches(['\r', '\n']).to_owned(),
        when: which.when(),
    })
}

/// The first message of `log` containing every needle.
fn find<'a>(log: &'a [Raw], needles: &[&str]) -> Res<&'a Raw> {
    log.iter()
        .find(|r| needles.iter().all(|n| r.message.contains(n)))
        .ok_or_else(|| format!("no message contains {needles:?}").into())
}

/// Messages per severity and type, in order of first appearance.
fn counts(log: &[Raw]) -> Res<Vec<Count>> {
    let mut out: Vec<Count> = Vec::new();
    for r in log {
        let (severity, kind) = (severity_name(r.severity)?, type_name(r.kind)?);
        match out
            .iter_mut()
            .find(|c| c.severity == severity && c.kind == kind)
        {
            Some(c) => c.n += 1,
            None => out.push(Count {
                severity,
                kind,
                n: 1,
            }),
        }
    }
    Ok(out)
}

/// Device name and driver version string of GPU 0, from a plain instance (no layers, so no messages).
fn gpu_label(entry: &Entry) -> Res<String> {
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_4);
    // SAFETY: the create info and the application info outlive the call.
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?
    };
    // SAFETY: `instance` is alive; the driver properties struct outlives the query.
    let label = unsafe { instance.enumerate_physical_devices() }
        .map_err(Into::into)
        .and_then(|gpus| -> Res<String> {
            let gpu = *gpus.first().ok_or("no Vulkan device")?;
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push(&mut driver);
            unsafe { instance.get_physical_device_properties2(gpu, &mut props) };
            let name = props
                .properties
                .device_name_as_c_str()?
                .to_string_lossy()
                .into_owned();
            Ok(format!(
                "{name}, driver {}",
                driver.driver_info_as_c_str()?.to_string_lossy()
            ))
        });
    // SAFETY: no child objects were created.
    unsafe { instance.destroy_instance(None) };
    label
}

/// Version of the installed Khronos validation layer (its spec version, e.g. 1.4.341).
fn layer_version(entry: &Entry) -> Res<String> {
    // SAFETY: plain query.
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    let v = layers
        .iter()
        .find(|l| {
            l.layer_name_as_c_str()
                .is_ok_and(|n| n == c"VK_LAYER_KHRONOS_validation")
        })
        .ok_or("VK_LAYER_KHRONOS_validation is not installed (install the Vulkan SDK)")?
        .spec_version;
    Ok(format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    ))
}

/// How many more messages an experiment produced than the base run (it only adds to step 02, never removes).
fn extra(run: &[Raw], base: &[Raw]) -> Res<usize> {
    run.len()
        .checked_sub(base.len())
        .ok_or_else(|| "an experiment produced fewer messages than the base run".into())
}

fn main() -> Res<()> {
    let out = out_path()?;
    // SAFETY: loads the system Vulkan loader; `entry` outlives every Vulkan object.
    let entry = unsafe { Entry::load()? };
    let source = format!(
        "Real messages of examples/rust step 02 on the course machine ({}, Khronos validation layer {}, {}), \
         captured by tools/vk-probe (binary validation-log, `bun run sync:validation-log`): the instance of step 02 \
         with a messenger that asks for every severity and type and records pMessageIdName. Lines marked \"when\": \
         \"leak\" come from the lesson's experiment (vkDestroyInstance with the messenger still alive); \"when\": \
         \"dab\" from the run that also asked for DEVICE_ADDRESS_BINDING_EXT.",
        gpu_label(&entry)?,
        layer_version(&entry)?,
        today()?,
    );
    let base = run(&entry, Run::Base)?;
    let leak = run(&entry, Run::Leak)?;
    let dab = run(&entry, Run::Dab)?;

    let mut messages = Vec::new();
    for (i, (which, needles)) in PICKS.iter().enumerate() {
        if i == LEAK_AT {
            for r in leak.iter().filter(|r| r.severity == Sev::ERROR) {
                messages.push(message(r, Run::Leak)?);
            }
        }
        let log = match which {
            Run::Base => &base,
            Run::Leak => &leak,
            Run::Dab => &dab,
        };
        messages.push(message(find(log, needles)?, *which)?);
    }

    let file = LogFile {
        source,
        run: Totals {
            total: base.len(),
            counts: counts(&base)?,
            extra: Extra {
                leak: extra(&leak, &base)?,
                dab: extra(&dab, &base)?,
            },
        },
        messages,
    };
    write_out(&out, &format!("{}\n", pretty(&file, 0)?))
}
