//! Stage feedback on stderr.
//!
//! stderr, not stdout: `--format json` has to stay pipeable into `jq` with the
//! progress lines still visible in the terminal.
//!
//! Redraws in place when stderr is a terminal, and falls back to one plain line
//! per stage when it is not, so a log file or CI transcript does not fill up
//! with carriage returns. Colour is dropped for a non-terminal and whenever
//! `NO_COLOR` is set.

use std::io::{IsTerminal, Write};
use std::time::Instant;

pub struct Progress {
    tty: bool,
    color: bool,
    start: Instant,
    step: Instant,
    open: bool,
}

impl Progress {
    pub fn new(enabled: bool) -> Self {
        let tty = enabled && std::io::stderr().is_terminal();
        Progress {
            tty,
            color: tty && std::env::var_os("NO_COLOR").is_none(),
            start: Instant::now(),
            step: Instant::now(),
            open: enabled,
        }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    /// Announce a stage. On a terminal this is overwritten by [`Progress::ok`].
    pub fn begin(&mut self, label: &str) {
        self.step = Instant::now();
        if !self.open {
            return;
        }
        let line = format!("  {} {label}...", self.paint("36", "*"));
        if self.tty {
            eprint!("\r\x1b[2K{line}");
        } else {
            eprintln!("{line}");
        }
        let _ = std::io::stderr().flush();
    }

    /// Close the current stage, reporting how long it took and a short note.
    pub fn ok(&mut self, label: &str, note: &str) {
        self.stage(label, self.step.elapsed().as_secs_f64(), note);
    }

    /// Report a stage whose duration was measured elsewhere. Stages that ran
    /// concurrently cannot share the single step clock, and their times
    /// overlap rather than summing to the total.
    pub fn stage(&mut self, label: &str, secs: f64, note: &str) {
        if !self.open {
            return;
        }
        let line = format!(
            "  {} {:<26} {:>7}  {}",
            self.paint("32", "v"),
            label,
            self.paint("2", &format!("{secs:.2}s")),
            self.paint("36", note),
        );
        if self.tty {
            eprintln!("\r\x1b[2K{line}");
        } else {
            eprintln!("{line}");
        }
    }

    /// Total wall time, once every stage is done.
    pub fn finish(&mut self) {
        if !self.open {
            return;
        }
        eprintln!(
            "  {} in {}\n",
            self.paint("2", "done"),
            self.paint("2", &format!("{:.2}s", self.start.elapsed().as_secs_f64()))
        );
    }
}
