//! Shared helpers of the probes that capture the course's real Vulkan data (see README.md).
//!
//! Each binary runs real Vulkan calls on the dev machine and writes one JSON file of `src/data/`, so the numbers and
//! messages the labs show are reproducible instead of copied by hand.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

pub type Res<T> = Result<T, Box<dyn Error>>;

/// The output path: the first command-line argument (the bun scripts pass the file in `src/data/`).
pub fn out_path() -> Res<PathBuf> {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| "usage: <probe> <output.json>".into())
}

/// Today's date in UTC as `YYYY-MM-DD` (the capture date stored in the files' metadata).
pub fn today() -> Res<String> {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(date_of(i64::try_from(secs / 86_400)?))
}

/// Civil date of a day number (days since 1970-01-01), from H. Hinnant's "chrono-Compatible Low-Level Date
/// Algorithms" (`civil_from_days`): std has no calendar, and one function does not justify a date crate.
fn date_of(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// `value` as pretty JSON (2-space indent, like the course's other data files), indented by `level` steps so it can
/// be nested in a hand-laid-out file.
pub fn pretty<T: Serialize>(value: &T, level: usize) -> Res<String> {
    let indent = format!("\n{}", "  ".repeat(level));
    Ok(serde_json::to_string_pretty(value)?.replace('\n', &indent))
}

/// Writes `text` to `path` through a temporary file in the same directory and a rename, so a failed run never
/// leaves a half-written data file behind.
pub fn write_out(path: &Path, text: &str) -> Res<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_of_known_days() {
        assert_eq!(super::date_of(0), "1970-01-01");
        assert_eq!(super::date_of(11_016), "2000-02-29");
        assert_eq!(super::date_of(20_736), "2026-10-10");
    }
}
