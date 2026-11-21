//! Cooperative cancellation of long computations, per operation.
//!
//! A [`Cancel`] token belongs to one operation (one training run, one forced alignment, one
//! corpus load, ...). The caller creates it, keeps a clone to [`Cancel::request`] cancellation
//! from another thread, and passes it to the core functions, which [`Cancel::check`] it at the
//! start of every training iteration, every phase task, every chunk of lines of the corpus
//! reader, the table setup, the subword projection and the alignment loops. When the token has
//! been requested, `check` unwinds with a [`Cancelled`] payload (via `resume_unwind`, so no
//! panic message is printed); the caller that started the work catches it with
//! `std::panic::catch_unwind` and looks for the payload with [`is_cancelled`]. Rayon carries
//! the payload from worker threads to the thread that started the parallel operation.
//!
//! Tokens are independent: requesting one never affects an operation holding another, and
//! there is nothing to reset between operations. [`Cancel::default`] is a token that nobody
//! else holds, i.e. an operation that cannot be cancelled. Checking a token never changes a
//! computed value.

use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cancellation token. Clones share the same request flag.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

/// Panic payload used to unwind a cancelled computation.
#[derive(Debug)]
pub struct Cancelled;

impl Cancel {
    /// A new token, not requested.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the operations holding this token (or a clone of it) to stop at their next check.
    pub fn request(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation of this token has been requested.
    pub fn is_requested(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Unwind with [`Cancelled`] if cancellation of this token was requested.
    #[inline]
    pub fn check(&self) {
        if self.is_requested() {
            cancelled();
        }
    }
}

#[cold]
#[inline(never)]
fn cancelled() -> ! {
    std::panic::resume_unwind(Box::new(Cancelled))
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
        let c = Cancel::new();
        assert!(catch_unwind(|| c.check()).is_ok());
        let clone = c.clone();
        clone.request();
        let r = catch_unwind(|| c.check());
        assert!(is_cancelled(&*r.unwrap_err()));
    }

    #[test]
    fn tokens_are_independent() {
        let (a, b) = (Cancel::new(), Cancel::new());
        a.request();
        assert!(a.is_requested());
        assert!(!b.is_requested());
        assert!(catch_unwind(|| b.check()).is_ok());
        assert!(!Cancel::default().is_requested());
    }
}
