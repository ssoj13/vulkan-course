//! Captures `VkMemoryRequirements` of buffers on GPU 0 into `src/data/buffer-requirements.json` (the buffer-memory
//! and memory-path labs): for every non-empty combination of six usage flags and five requested sizes it creates a
//! buffer, asks for its requirements and destroys it.
//!
//! `bun run sync:buffers` (or `cargo run --release --bin buffer-requirements -- <output.json>`).

use ash::{Entry, vk};
use serde::Serialize;
use vk_probe::{Res, out_path, pretty, today, write_out};

/// Usage flags of the capture, in bit order of the combination mask (row `mask` has flag `i` when bit `i` is set).
const FLAGS: [(&str, vk::BufferUsageFlags); 6] = [
    (
        "VK_BUFFER_USAGE_TRANSFER_SRC_BIT",
        vk::BufferUsageFlags::TRANSFER_SRC,
    ),
    (
        "VK_BUFFER_USAGE_TRANSFER_DST_BIT",
        vk::BufferUsageFlags::TRANSFER_DST,
    ),
    (
        "VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT",
        vk::BufferUsageFlags::UNIFORM_BUFFER,
    ),
    (
        "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT",
        vk::BufferUsageFlags::STORAGE_BUFFER,
    ),
    (
        "VK_BUFFER_USAGE_INDEX_BUFFER_BIT",
        vk::BufferUsageFlags::INDEX_BUFFER,
    ),
    (
        "VK_BUFFER_USAGE_VERTEX_BUFFER_BIT",
        vk::BufferUsageFlags::VERTEX_BUFFER,
    ),
];

/// Requested sizes, bytes: tiny, unaligned, a page-ish size, 64 KiB and an odd megabyte, to expose the rounding.
const REQUESTS: [u64; 5] = [1, 100, 1000, 65536, 1_000_003];

const SOURCE: &str = "Real capture on the course dev machine (same GPU and driver as gpu.json) by tools/vk-probe \
    (binary buffer-requirements, `bun run sync:buffers`; ash at the course pin): it creates a Vulkan 1.4 instance and \
    a device on GPU 0 (queue family 0, no features), then for every non-empty combination of the six usage flags seen \
    in `buffers` (63 rows) and each requested size calls vkCreateBuffer (sharingMode EXCLUSIVE, no flags), \
    vkGetBufferMemoryRequirements and vkDestroyBuffer. memoryTypePropertyFlags: the raw VkMemoryPropertyFlags of the \
    memory types the same run saw (must match gpu.json).";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Meta {
    source: &'static str,
    device: String,
    driver_version_raw: u32,
    date: String,
    memory_type_property_flags: Vec<u32>,
    requests: [u64; 5],
}

/// One usage combination. Alignment and memoryTypeBits do not depend on the size (checked), so they are stored once.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Row {
    usage: Vec<&'static str>,
    alignment: u64,
    memory_type_bits: u32,
    sizes: Vec<u64>,
}

fn main() -> Res<()> {
    let out = out_path()?;
    // SAFETY: loads the system Vulkan loader; `entry` outlives every Vulkan object below.
    let entry = unsafe { Entry::load()? };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_4);
    // SAFETY: the create info and the application info it points to outlive the call.
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?
    };
    let res = capture(&instance);
    // SAFETY: every child object was destroyed inside `capture`, on success and on error.
    unsafe { instance.destroy_instance(None) };
    let (meta, rows) = res?;

    let rows: Vec<String> = rows
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()?;
    let text = format!(
        "{{\n  \"meta\": {},\n  \"buffers\": [\n    {}\n  ]\n}}\n",
        pretty(&meta, 1)?,
        rows.join(",\n    ")
    );
    write_out(&out, &text)
}

/// Creates the device on GPU 0, runs the capture, and destroys the device whatever the outcome.
fn capture(instance: &ash::Instance) -> Res<(Meta, Vec<Row>)> {
    // SAFETY: `instance` is alive.
    let gpu = *unsafe { instance.enumerate_physical_devices()? }
        .first()
        .ok_or("no Vulkan device")?;
    // SAFETY: `gpu` was just enumerated from `instance`.
    let (props, mem) = unsafe {
        (
            instance.get_physical_device_properties(gpu),
            instance.get_physical_device_memory_properties(gpu),
        )
    };
    let prio = [1.0f32];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&prio)];
    // SAFETY: family 0 exists on every device (a device has at least one queue family); the infos outlive the call.
    let device = unsafe {
        instance.create_device(
            gpu,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
            None,
        )?
    };
    let rows = (1u32..1 << FLAGS.len())
        .map(|mask| row(&device, mask))
        .collect::<Res<Vec<_>>>();
    // SAFETY: every buffer was destroyed in `requirements`; nothing else uses the device.
    unsafe { device.destroy_device(None) };

    let types = mem.memory_types_as_slice();
    let meta = Meta {
        source: SOURCE,
        device: props.device_name_as_c_str()?.to_string_lossy().into_owned(),
        driver_version_raw: props.driver_version,
        date: today()?,
        memory_type_property_flags: types.iter().map(|t| t.property_flags.as_raw()).collect(),
        requests: REQUESTS,
    };
    Ok((meta, rows?))
}

/// The row of usage combination `mask`: requirements at every requested size.
fn row(device: &ash::Device, mask: u32) -> Res<Row> {
    let picked: Vec<_> = FLAGS
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, f)| *f)
        .collect();
    let usage = picked
        .iter()
        .fold(vk::BufferUsageFlags::empty(), |u, (_, f)| u | *f);
    let names: Vec<&'static str> = picked.iter().map(|(n, _)| *n).collect();
    let reqs = REQUESTS
        .iter()
        .map(|&size| requirements(device, size, usage))
        .collect::<Res<Vec<_>>>()?;
    let first = reqs.first().ok_or("no requested sizes")?;
    if reqs
        .iter()
        .any(|r| r.alignment != first.alignment || r.memory_type_bits != first.memory_type_bits)
    {
        return Err(format!("alignment or memoryTypeBits vary with the size for {names:?}").into());
    }
    Ok(Row {
        usage: names,
        alignment: first.alignment,
        memory_type_bits: first.memory_type_bits,
        sizes: reqs.iter().map(|r| r.size).collect(),
    })
}

/// vkCreateBuffer -> vkGetBufferMemoryRequirements -> vkDestroyBuffer.
fn requirements(
    device: &ash::Device,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Res<vk::MemoryRequirements> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `device` is alive and `info` is a valid create info (non-zero size, non-empty usage).
    unsafe {
        let buffer = device.create_buffer(&info, None)?;
        let reqs = device.get_buffer_memory_requirements(buffer);
        device.destroy_buffer(buffer, None);
        Ok(reqs)
    }
}
