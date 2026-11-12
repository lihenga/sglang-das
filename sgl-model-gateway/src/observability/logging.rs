//! Logging infrastructure with non-blocking file I/O.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tracing::Level;
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{RollingFileAppender, Rotation},
};
use tracing_log::LogTracer;
use tracing_subscriber::{
    fmt::time::ChronoUtc, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer,
};

use super::otel_trace::get_otel_layer;
use crate::config::TraceConfig;

const TIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
const TIME_FORMAT_MS: &str = "%Y-%m-%d %H:%M:%S%.3f";
const DEFAULT_LOG_TARGET: &str = "smg";

pub(crate) struct LogRateLimiter {
    state: Mutex<(Option<Instant>, u64)>,
}

impl LogRateLimiter {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new((None, 0)),
        }
    }

    pub(crate) fn check(&self) -> Option<u64> {
        self.check_at(Instant::now())
    }

    fn check_at(&self, now: Instant) -> Option<u64> {
        let mut state = self.state.lock();
        if state
            .0
            .is_some_and(|last| now.saturating_duration_since(last) < Duration::from_secs(30))
        {
            state.1 = state.1.saturating_add(1);
            return None;
        }
        state.0 = Some(now);
        Some(std::mem::take(&mut state.1))
    }
}

// One process-wide limiter per call site; never throttle the surrounding error handling.
macro_rules! rate_limited_log {
    ($level:ident, $($fields:tt)*) => {{
        static LIMITER: $crate::observability::logging::LogRateLimiter =
            $crate::observability::logging::LogRateLimiter::new();
        if let Some(suppressed_logs) = LIMITER.check() {
            tracing::event!(tracing::Level::$level, suppressed_logs, $($fields)*);
        }
    }};
}

pub(crate) use rate_limited_log;

fn get_time_format() -> &'static str {
    match std::env::var("SGLANG_LOG_MS") {
        Ok(v) if matches!(v.trim().to_lowercase().as_str(), "true" | "1") => TIME_FORMAT_MS,
        _ => TIME_FORMAT,
    }
}

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub level: Level,
    pub json_format: bool,
    pub log_dir: Option<String>,
    pub colorize: bool,
    pub log_file_name: String,
    pub log_targets: Option<Vec<String>>,
}

impl Default for LoggingConfig {
    #[inline]
    fn default() -> Self {
        Self {
            level: Level::INFO,
            json_format: false,
            log_dir: None,
            colorize: true,
            log_file_name: "smg".to_string(),
            log_targets: Some(vec![DEFAULT_LOG_TARGET.to_string()]),
        }
    }
}

/// Guard that keeps the file appender thread alive.
#[allow(dead_code)]
pub struct LogGuard {
    _file_guard: Option<WorkerGuard>,
}

#[inline]
const fn level_to_str(level: Level) -> &'static str {
    match level {
        Level::TRACE => "trace",
        Level::DEBUG => "debug",
        Level::INFO => "info",
        Level::WARN => "warn",
        Level::ERROR => "error",
    }
}

#[inline]
fn build_filter_string(targets: &[String], level_filter: &str) -> String {
    // Exact capacity: sum of target lengths + "=" and level per target + commas between
    let capacity = targets.iter().map(String::len).sum::<usize>()
        + targets.len() * (1 + level_filter.len())
        + targets.len().saturating_sub(1);
    let mut filter_string = String::with_capacity(capacity);

    for (i, target) in targets.iter().enumerate() {
        if i > 0 {
            filter_string.push(',');
        }
        filter_string.push_str(target);
        filter_string.push('=');
        filter_string.push_str(level_filter);
    }

    filter_string
}

pub fn init_logging(config: LoggingConfig, otel_layer_config: Option<TraceConfig>) -> LogGuard {
    let _ = LogTracer::init();

    let level_filter = level_to_str(config.level);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let filter_string = match &config.log_targets {
            Some(targets) if !targets.is_empty() => build_filter_string(targets, level_filter),
            _ => {
                let mut s =
                    String::with_capacity(DEFAULT_LOG_TARGET.len() + 1 + level_filter.len());
                s.push_str(DEFAULT_LOG_TARGET);
                s.push('=');
                s.push_str(level_filter);
                s
            }
        };

        EnvFilter::new(filter_string)
    });

    let mut layers = Vec::with_capacity(3);

    let time_fmt = get_time_format();

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_ansi(config.colorize)
        .with_file(true)
        .with_line_number(true)
        .with_timer(ChronoUtc::new(time_fmt.to_string()));

    let stdout_layer = if config.json_format {
        stdout_layer.json().flatten_event(true).boxed()
    } else {
        stdout_layer.boxed()
    };

    layers.push(stdout_layer);

    let mut file_guard = None;

    if let Some(log_dir) = &config.log_dir {
        let log_dir = PathBuf::from(log_dir);

        if !log_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&log_dir) {
                eprintln!("Failed to create log directory: {}", e);
                return LogGuard { _file_guard: None };
            }
        }

        let file_appender =
            RollingFileAppender::new(Rotation::DAILY, log_dir, &config.log_file_name);

        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        file_guard = Some(guard);

        let file_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_file(true)
            .with_line_number(true)
            .with_timer(ChronoUtc::new(time_fmt.to_string()))
            .with_writer(non_blocking);

        let file_layer = if config.json_format {
            file_layer.json().flatten_event(true).boxed()
        } else {
            file_layer.boxed()
        };

        layers.push(file_layer);
    }

    if let Some(otel_layer_config) = &otel_layer_config {
        if otel_layer_config.enable_trace {
            match get_otel_layer() {
                Ok(otel_layer) => {
                    layers.push(otel_layer);
                }
                Err(e) => {
                    eprintln!("Failed to initialize OpenTelemetry: {}", e);
                }
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .try_init();

    LogGuard {
        _file_guard: file_guard,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{self, Write},
        sync::{Arc, Barrier},
    };

    use super::*;

    #[test]
    fn limiter_emits_first_event_and_counts_suppressed_events_per_window() {
        let limiter = LogRateLimiter::new();
        let now = Instant::now();
        assert_eq!(limiter.check_at(now), Some(0));
        assert_eq!(limiter.check_at(now), None);
        assert_eq!(limiter.check_at(now + Duration::from_secs(29)), None);
        assert_eq!(limiter.check_at(now + Duration::from_secs(30)), Some(2));
        assert_eq!(limiter.check_at(now + Duration::from_secs(59)), None);
        assert_eq!(limiter.check_at(now + Duration::from_secs(60)), Some(1));
    }

    #[test]
    fn concurrent_failures_share_the_same_budget() {
        let limiter = Arc::new(LogRateLimiter::new());
        let barrier = Arc::new(Barrier::new(16));
        let now = Instant::now();
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let limiter = limiter.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    limiter.check_at(now)
                })
            })
            .collect();
        let emitted = threads
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .count();
        assert_eq!(emitted, 1);
        assert_eq!(limiter.check_at(now + Duration::from_secs(30)), Some(15));
    }

    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn macro_limits_each_call_site_across_different_requests_and_preserves_level() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let writer = Capture(output.clone());
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            for request in 0..20 {
                rate_limited_log!(WARN, request, "limited warning");
                rate_limited_log!(ERROR, request, "limited error");
            }
        });
        let bytes = output.lock();
        let logs = String::from_utf8_lossy(&bytes);
        assert_eq!(logs.lines().count(), 2, "{logs}");
        assert!(logs.contains("WARN"));
        assert!(logs.contains("ERROR"));
        assert!(logs.contains("limited warning"));
        assert!(logs.contains("limited error"));
        assert_eq!(logs.matches("suppressed_logs=0").count(), 2, "{logs}");
    }
}
