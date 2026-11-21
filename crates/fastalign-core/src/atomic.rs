//! Crash-safe file output: write to a private temporary file next to the destination, then
//! replace the destination in one step.
//!
//! Readers see the old file or the complete new one, never a partial file; an error or a drop
//! before [`AtomicFile::commit`] leaves the destination untouched and removes the temporary
//! file. The temporary name is unique per call (process id, a counter and the clock) and is
//! created with `create_new`, so it can never clobber a file that already exists, such as an
//! unrelated `<destination>.tmp`.
//!
//! A destination that exists but is not a regular file (`/dev/null`, a named pipe, a terminal)
//! cannot be replaced by renaming, so it is written directly.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

enum Target {
    /// Staged in `tmp`; renamed over `dest` by `commit`.
    Staged { tmp: PathBuf, dest: PathBuf },
    /// A special file that is written in place.
    Direct,
}

/// A buffered writer whose contents replace `dest` only when [`commit`](Self::commit) succeeds.
pub struct AtomicFile {
    w: Option<BufWriter<File>>,
    target: Target,
}

impl AtomicFile {
    /// Start writing a replacement for `dest`.
    pub fn create(dest: &Path) -> io::Result<Self> {
        match fs::metadata(dest) {
            Ok(m) if !m.is_file() => {
                let f = OpenOptions::new().write(true).open(dest)?;
                return Ok(Self {
                    w: Some(BufWriter::with_capacity(1 << 20, f)),
                    target: Target::Direct,
                });
            }
            _ => {}
        }
        let dir = match dest.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let name = dest
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "not a file path"))?;
        let mut last = None;
        for _ in 0..100 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            let mut n = name.to_os_string();
            n.push(format!(
                ".{}-{}-{nanos:x}.tmp",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let tmp = dir.join(n);
            match OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(f) => {
                    return Ok(Self {
                        w: Some(BufWriter::with_capacity(1 << 20, f)),
                        target: Target::Staged {
                            tmp,
                            dest: dest.to_path_buf(),
                        },
                    })
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| io::Error::other("no free temporary name")))
    }

    /// Flush, sync and publish. On error the destination is unchanged.
    pub fn commit(mut self) -> io::Result<()> {
        let w = self.w.take().expect("open");
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all().or_else(|e| match self.target {
            // fsync is meaningless on some special files (terminals, pipes)
            Target::Direct => Ok(()),
            Target::Staged { .. } => Err(e),
        })?;
        drop(f);
        if let Target::Staged { tmp, dest } = &self.target {
            // std's rename replaces an existing file on Windows as well
            fs::rename(tmp, dest)?;
            self.target = Target::Direct; // nothing left for drop to remove
        }
        Ok(())
    }
}

impl Write for AtomicFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.w.as_mut().expect("open").write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.w.as_mut().expect("open").flush()
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if let Target::Staged { tmp, .. } = &self.target {
            self.w = None; // close first (required on Windows)
            let _ = fs::remove_file(tmp);
        }
    }
}

/// Atomically replace `path` with `bytes`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = AtomicFile::create(path)?;
    f.write_all(bytes)?;
    f.commit()
}

/// Atomically replace `path` with what `fill` writes; an error from `fill` leaves `path` alone.
pub fn write_with<E: From<io::Error>>(
    path: &Path,
    fill: impl FnOnce(&mut dyn Write) -> Result<(), E>,
) -> Result<(), E> {
    let mut f = AtomicFile::create(path)?;
    fill(&mut f)?;
    f.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_and_keeps_sidecar() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("checkpoint");
        let side = d.path().join("checkpoint.tmp");
        fs::write(&side, b"other").unwrap();
        fs::write(&p, b"old").unwrap();
        write_atomic(&p, b"new").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"new");
        assert_eq!(fs::read(&side).unwrap(), b"other");
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), 2);
    }

    #[test]
    fn failure_and_drop_keep_old_file() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("m");
        fs::write(&p, b"good").unwrap();
        let r: Result<(), io::Error> = write_with(&p, |w| {
            w.write_all(b"partial")?;
            Err(io::Error::other("boom"))
        });
        assert!(r.is_err());
        assert_eq!(fs::read(&p).unwrap(), b"good");
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), 1);
        let mut f = AtomicFile::create(&p).unwrap();
        f.write_all(b"x").unwrap();
        drop(f);
        assert_eq!(fs::read(&p).unwrap(), b"good");
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), 1);
    }

    #[test]
    fn missing_directory_is_an_error_and_special_files_work() {
        let d = tempfile::tempdir().unwrap();
        assert!(write_atomic(&d.path().join("no/such/x"), b"a").is_err());
        #[cfg(unix)]
        write_atomic(Path::new("/dev/null"), b"a").unwrap();
    }

    #[test]
    fn concurrent_writers_leave_one_complete_file() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("m");
        let hs: Vec<_> = (0..8u8)
            .map(|i| {
                let p = p.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        write_atomic(&p, &vec![i; 1 << 20]).unwrap();
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let b = fs::read(&p).unwrap();
        assert_eq!(b.len(), 1 << 20);
        assert!(b.iter().all(|&x| x == b[0]));
        assert_eq!(fs::read_dir(d.path()).unwrap().count(), 1);
    }
}
