//! Shared streaming helpers used by the HTTP / OpenAI / PD routers.
//!
//! The main type here is [`BreakerTrackedStream`], a `Stream` adapter that
//! records circuit-breaker outcomes based on how the upstream stream
//! terminates:
//!
//! - **Clean end** (`Poll::Ready(None)`) → `record_success`.
//! - **Upstream transport error** (`Poll::Ready(Some(Err(_)))`) → `record_failure`.
//! - **Caller drops the stream while still active** (client disconnect) → no
//!   breaker call. The HTTP response already shipped a 200 status; whether
//!   the worker is healthy is unknown from this signal alone.
//!
//! Callers can override the terminal state in two cases:
//!
//! - [`BreakerTrackedStream::mark_completed`] — for routers that detect
//!   end-of-stream via an in-band sentinel (e.g. PD's `data: [DONE]`)
//!   before the underlying byte stream returns `None`. Note that
//!   `Completed` is *not* absorbing — a subsequent `poll_next` returning
//!   `Err` will still escalate the terminal to `Errored`. Callers should
//!   stop polling after `mark_completed`.
//! - [`BreakerTrackedStream::mark_errored`] — for routers wrapping content
//!   that already represents a worker failure (e.g. a non-2xx response
//!   body or a synthetic SSE error envelope built from a 5xx). `Errored`
//!   is absorbing: once set, it persists regardless of later polls.

use std::{
    fmt::Display,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use bytes::Bytes;
use futures_util::Stream;
use tracing::error;

use crate::{core::Worker, observability::metrics::Metrics};

/// Only retain the prefix needed to recognize the engine's error envelope.
#[derive(Default)]
struct ErrorEventScanner {
    line_start: Vec<u8>,
    saw_error: bool,
}

impl ErrorEventScanner {
    fn feed(&mut self, chunk: &[u8]) {
        if self.saw_error {
            return;
        }
        // shortcut: inspect at most 32 bytes per line for the engine's first-key
        // error envelope; use a streaming parser if other SSE formats are needed.
        const LINE_PROBE: usize = 32;
        for (i, segment) in chunk.split(|&b| b == b'\n').enumerate() {
            if i > 0 {
                self.saw_error |= Self::is_error_line(&self.line_start);
                self.line_start.clear();
            }
            let room = LINE_PROBE - self.line_start.len();
            self.line_start
                .extend_from_slice(&segment[..segment.len().min(room)]);
            self.saw_error |= Self::is_error_line(&self.line_start);
        }
    }

    fn is_error_line(line: &[u8]) -> bool {
        line.strip_prefix(b"data:")
            .map(|p| p.trim_ascii_start())
            .and_then(|p| p.strip_prefix(b"{"))
            .map(|p| p.trim_ascii_start())
            .is_some_and(|p| p.starts_with(b"\"error\""))
    }
}

struct StreamMetrics {
    backend_type: &'static str,
    model: String,
    scanner: ErrorEventScanner,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terminal {
    /// Stream is still in flight, or was dropped before terminating.
    Active,
    /// Stream returned `None` or the caller marked it complete.
    Completed,
    /// Stream yielded an `Err` item.
    Errored,
}

/// Wraps a `Stream<Item = Result<Bytes, E>>` so that the circuit breaker on
/// `worker` is updated exactly once on drop:
/// - completed → `record_success`
/// - errored → `record_failure`
/// - dropped while still active → neither (client disconnected; the worker
///   is innocent from our point of view).
///
/// `E` defaults to `reqwest::Error` to match the common producer
/// (`response.bytes_stream()`), but can be any `Display + Send + 'static`
/// error type — useful for tests that construct streams with simpler
/// error types.
#[must_use = "BreakerTrackedStream must be polled to completion (or pre-marked) \
              and then dropped for the circuit breaker to record an outcome; \
              discarding it immediately records nothing"]
pub struct BreakerTrackedStream<E = reqwest::Error> {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, E>> + Send + 'static>>,
    worker: Arc<dyn Worker>,
    log_url: String,
    terminal: Terminal,
    metrics: Option<StreamMetrics>,
}

impl<E> BreakerTrackedStream<E> {
    pub fn new<S>(inner: S, worker: Arc<dyn Worker>, log_url: String) -> Self
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    {
        Self {
            inner: Box::pin(inner),
            worker,
            log_url,
            terminal: Terminal::Active,
            metrics: None,
        }
    }

    /// Enable observation only after the caller has received 2xx headers.
    pub fn with_metrics(mut self, backend_type: &'static str, model: &str) -> Self {
        self.metrics = Some(StreamMetrics {
            backend_type,
            model: model.to_owned(),
            scanner: ErrorEventScanner::default(),
        });
        self
    }

    fn stream_outcome(&self) -> &'static str {
        if self.terminal == Terminal::Errored {
            "upstream_error"
        } else if self.metrics.as_ref().is_some_and(|m| m.scanner.saw_error) {
            "stream_error_event"
        } else if self.terminal == Terminal::Active {
            "client_disconnect"
        } else {
            "ok"
        }
    }

    /// Mark the stream as cleanly completed. Use this from callers that
    /// detect end-of-stream via an in-band sentinel (e.g. `data: [DONE]`)
    /// before the underlying byte stream returns `None`.
    ///
    /// Has no effect once the wrapper is in any non-Active state. `Completed`
    /// is *not* absorbing — a later `poll_next` returning `Err` will still
    /// escalate the terminal to `Errored`. Callers should stop polling
    /// after calling this.
    pub fn mark_completed(&mut self) {
        if self.terminal == Terminal::Active {
            self.terminal = Terminal::Completed;
        }
    }

    /// Pre-tag the stream as terminally errored. Use this from callers
    /// constructing a wrapper around content that already represents a
    /// failed worker outcome (e.g. a non-2xx response body or a
    /// synthetic error envelope) so Drop records `record_failure` even
    /// if the underlying stream terminates cleanly. `Errored` is
    /// absorbing — once set it stays set regardless of later events.
    pub fn mark_errored(&mut self) {
        self.terminal = Terminal::Errored;
    }
}

impl<E: Display> Stream for BreakerTrackedStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => {
                if let Some(metrics) = self.metrics.as_mut() {
                    metrics.scanner.feed(&b);
                }
                Poll::Ready(Some(Ok(b)))
            }
            Poll::Ready(Some(Err(e))) => {
                error!("Upstream stream error from worker {}: {}", self.log_url, e);
                self.terminal = Terminal::Errored;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if self.terminal == Terminal::Active {
                    self.terminal = Terminal::Completed;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<E> Drop for BreakerTrackedStream<E> {
    fn drop(&mut self) {
        match self.terminal {
            Terminal::Completed => self.worker.circuit_breaker().record_success(),
            Terminal::Errored => self.worker.circuit_breaker().record_failure(),
            // Client disconnected before we knew the worker's verdict.
            // Leaving the breaker untouched is the correct default — we
            // got a 200 header and some bytes; nothing said the worker
            // is unhealthy.
            Terminal::Active => {}
        }
        if let Some(metrics) = &self.metrics {
            Metrics::record_http_stream_outcome(
                metrics.backend_type,
                &metrics.model,
                self.worker.url(),
                self.stream_outcome(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt, sync::Arc};

    use bytes::Bytes;
    use futures_util::StreamExt;

    use super::{BreakerTrackedStream, ErrorEventScanner};
    use crate::core::{BasicWorkerBuilder, Worker};

    fn drop_and_render_metrics<E>(tracked: BreakerTrackedStream<E>) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || drop(tracked));
        handle.render()
    }

    fn assert_outcome(metrics: &str, outcome: &str) {
        let lines: Vec<_> = metrics
            .lines()
            .filter(|line| line.starts_with("smg_router_stream_outcome_total{"))
            .collect();
        assert_eq!(lines.len(), 1, "{metrics}");
        assert!(lines[0].contains(&format!("outcome=\"{outcome}\"")));
        assert!(lines[0].ends_with(" 1"), "{metrics}");
    }

    /// Lightweight error type for tests — keeps the wrapper generic so we
    /// don't need to fabricate `reqwest::Error` instances.
    #[derive(Debug)]
    struct TestErr(&'static str);

    impl fmt::Display for TestErr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://test-worker").build())
    }

    fn breaker_counters(w: &Arc<dyn Worker>) -> (u64, u64) {
        let cb = w.circuit_breaker();
        (cb.total_successes(), cb.total_failures())
    }

    #[tokio::test]
    async fn drop_while_active_records_disconnect_without_breaker_outcome() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("regular", "test-model");
        assert_outcome(&drop_and_render_metrics(tracked), "client_disconnect");
        assert_eq!(breaker_counters(&w), (0, 0));
    }

    #[tokio::test]
    async fn clean_stream_records_one_success() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("regular", "test-model");
        while tracked.next().await.is_some() {}
        assert_outcome(&drop_and_render_metrics(tracked), "ok");
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn stream_error_records_one_failure() {
        let w = worker();
        let inner =
            futures_util::stream::iter(vec![Ok(Bytes::from_static(b"a")), Err(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("regular", "test-model");
        while tracked.next().await.is_some() {}
        assert_outcome(&drop_and_render_metrics(tracked), "upstream_error");
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn errored_is_absorbing_across_subsequent_polls() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Err::<Bytes, _>(TestErr("boom")),
            Ok(Bytes::from_static(b"after")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_then_drop_records_success() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("pd", "test-model");
        tracked.mark_completed();
        assert_outcome(&drop_and_render_metrics(tracked), "ok");
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn mark_errored_then_clean_end_still_records_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Ok::<_, TestErr>(Bytes::from_static(b"a"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        tracked.mark_errored();
        while tracked.next().await.is_some() {}
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[tokio::test]
    async fn mark_completed_does_not_overwrite_errored() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![Err::<Bytes, _>(TestErr("boom"))]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        while tracked.next().await.is_some() {}
        tracked.mark_completed();
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    // PD's [DONE] handler calls mark_completed before the underlying byte
    // stream finishes; a trailing transport error must still flip the
    // terminal to Errored so the breaker records failure.
    #[tokio::test]
    async fn mark_completed_then_later_err_escalates_to_failure() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok::<_, TestErr>(Bytes::from_static(b"data: [DONE]\n\n")),
            Err(TestErr("trailing transport error")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into());
        // Caller observes the [DONE] chunk and pre-marks completed...
        assert!(tracked.next().await.is_some());
        tracked.mark_completed();
        // ...but a trailing poll surfaces a transport error.
        assert!(matches!(tracked.next().await, Some(Err(_))));
        drop(tracked);
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[test]
    fn scanner_handles_chunk_boundaries_without_matching_content() {
        let mut scanner = ErrorEventScanner::default();
        scanner.feed(b"data: {\"choices\":[{\"content\":\"data: {\\\"error\\\"\"}]}\n\n");
        assert!(!scanner.saw_error);
        scanner.feed(b"data:{ \"err");
        scanner.feed(b"or\":{\"code\":503}}\n\n");
        assert!(scanner.saw_error);

        let mut scanner = ErrorEventScanner::default();
        scanner.feed(&vec![b'x'; 1024 * 1024]);
        assert_eq!(scanner.line_start.len(), 32);
        assert!(!scanner.saw_error);
    }

    #[test]
    fn scanner_recognizes_error_before_line_terminator() {
        let mut scanner = ErrorEventScanner::default();
        scanner.feed(b"data: {\"err");
        assert!(!scanner.saw_error);
        scanner.feed(b"or\":{\"code\":503}}");
        assert!(scanner.saw_error);
    }

    #[tokio::test]
    async fn in_band_error_preserves_bytes_and_breaker_behavior() {
        let w = worker();
        let chunks = vec![
            Bytes::from_static(b"data: {\"err"),
            Bytes::from_static(b"or\":{\"code\":503}}\n\ndata: [DONE]\n\n"),
        ];
        let inner = futures_util::stream::iter(
            chunks
                .iter()
                .cloned()
                .map(Ok::<_, TestErr>)
                .collect::<Vec<_>>(),
        );
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("pd", "requested-model");
        let mut received = Vec::new();
        while let Some(chunk) = tracked.next().await {
            received.push(chunk.unwrap());
        }
        assert_eq!(received, chunks);
        let metrics = drop_and_render_metrics(tracked);
        assert_outcome(&metrics, "stream_error_event");
        assert!(metrics.contains("backend_type=\"pd\""));
        assert!(metrics.contains("model=\"requested-model\""));
        assert!(metrics.contains("worker_url=\"http://test-worker\""));
        assert_eq!(breaker_counters(&w), (1, 0));
    }

    #[tokio::test]
    async fn transport_error_outranks_in_band_error() {
        let w = worker();
        let inner = futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"data: {\"error\":{\"code\":503}}\n\n")),
            Err(TestErr("reset")),
        ]);
        let mut tracked = BreakerTrackedStream::new(inner, Arc::clone(&w), "u".into())
            .with_metrics("regular", "test-model");
        while tracked.next().await.is_some() {}
        assert_outcome(&drop_and_render_metrics(tracked), "upstream_error");
        assert_eq!(breaker_counters(&w), (0, 1));
    }

    #[test]
    fn streams_without_metrics_do_not_create_series() {
        let w = worker();
        let inner = futures_util::stream::pending::<Result<Bytes, TestErr>>();
        let tracked = BreakerTrackedStream::new(inner, w, "u".into());
        assert!(!drop_and_render_metrics(tracked).contains("smg_router_stream_outcome_total"));
    }
}
