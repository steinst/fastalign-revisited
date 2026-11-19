//! Cooperative cancellation of long computations.
//!
//! A process-wide flag, checked at the start of every training iteration, every phase task and
//! every progress event. When it is set, [`check`] unwinds with a [`Cancelled`] payload (via
//! `resume_unwind`, so no panic message is printed); callers that started the work catch it with
//! `std::panic::catch_unwind` and look for the payload with [`is_cancelled`]. Rayon carries the
//! payload from worker threads to the thread that started the parallel operation.
//!
//! The flag is global: cancelling stops every computation in the process, and a new top-level
//! call should [`reset`] it first. Checking the flag never changes a computed value.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};

static FLAG: AtomicBool = AtomicBool::new(false);

/// Panic payload used to unwind a cancelled computation.
#[derive(Debug)]
pub struct Cancelled;

/// Ask running computations to stop at their next check.
pub fn request() {
    FLAG.store(true, Ordering::Relaxed);
}

/// Clear a pending request.
pub fn reset() {
    FLAG.store(false, Ordering::Relaxed);
}

/// Whether cancellation has been requested.
pub fn is_requested() -> bool {
    FLAG.load(Ordering::Relaxed)
}

/// Unwind with [`Cancelled`] if cancellation was requested.
#[inline]
pub fn check() {
    if FLAG.load(Ordering::Relaxed) {
        std::panic::resume_unwind(Box::new(Cancelled));
    }
}

/// Whether a caught panic payload is a [`Cancelled`] unwind.
pub fn is_cancelled(payload: &(dyn Any + Send)) -> bool {
    payload.is::<Cancelled>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::catch_unwind;

    #[test]
    fn check_unwinds_only_when_requested() {
        reset();
        assert!(catch_unwind(check).is_ok());
        request();
        let r = catch_unwind(check);
        reset();
        assert!(is_cancelled(&*r.unwrap_err()));
    }
}
