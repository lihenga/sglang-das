use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
};

use axum::{routing::post, Json, Router};
use eventsource_stream::Eventsource;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::{oneshot, Notify},
};

use super::{tests::create_test_pd_router, *};
use crate::core::{BasicWorkerBuilder, CircuitBreakerConfig};

struct TestServer {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(app: Router) -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    TestServer { url, task }
}

// A raw peer lets us verify that cancellation really closes the upstream socket.
async fn pending_peer() -> (TestServer, Arc<Notify>, oneshot::Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let started = Arc::new(Notify::new());
    let ready = started.clone();
    let (closed_tx, closed_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 8192];
        assert!(socket.read(&mut buf).await.unwrap() > 0);
        ready.notify_one();
        let _ = socket.read_to_end(&mut Vec::new()).await;
        let _ = closed_tx.send(());
    });
    (TestServer { url, task }, started, closed_rx)
}

fn worker(url: &str, kind: WorkerType) -> Arc<dyn Worker> {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(kind)
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        })
        .build();
    worker.set_healthy(true);
    Arc::new(worker)
}

fn context(stream: bool) -> PDRequestContext<'static> {
    PDRequestContext {
        route: "/generate",
        batch_size: None,
        is_stream: stream,
        return_logprob: false,
        request_text: None,
        model_id: None,
        headers: None,
    }
}

async fn dispatch(p: &str, d: &str, stream: bool) -> Response {
    dispatch_with_limit(
        p,
        d,
        stream,
        crate::config::RouterConfig::default().max_payload_size,
    )
    .await
}

async fn dispatch_with_limit(p: &str, d: &str, stream: bool, limit: usize) -> Response {
    let mut router = create_test_pd_router();
    router.max_payload_size = limit;
    tokio::time::timeout(
        Duration::from_secs(2),
        router.execute_dual_dispatch_internal(
            None,
            json!({"text":"prompt", "stream":stream}),
            context(stream),
            worker(
                p,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            worker(d, WorkerType::Decode),
            tokio::time::Instant::now() + Duration::from_secs(10),
        ),
    )
    .await
    .expect("failed pair must not wait for the other worker")
}

async fn observe_metrics<F: Future>(future: F) -> (F::Output, String) {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    tokio::pin!(future);
    let output = std::future::poll_fn(|cx| {
        metrics::with_local_recorder(&recorder, || future.as_mut().poll(cx))
    })
    .await;
    (output, handle.render())
}

#[tokio::test]
async fn decode_http_failure_cancels_pending_prefill_even_if_error_body_stalls() {
    for stream in [false, true] {
        let (prefill, started, closed) = pending_peer().await;
        let decode = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Body::from_stream(futures_util::stream::pending::<
                            Result<bytes::Bytes, std::io::Error>,
                        >()),
                    )
                }
            }),
        ))
        .await;
        let response = dispatch(&prefill.url, &decode.url, stream).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn decode_transport_failure_does_not_wait_for_prefill() {
    let (prefill, _, _) = pending_peer().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let absent = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let response = dispatch(&prefill.url, &absent, true).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!String::from_utf8_lossy(&body).contains(&absent));
}

#[tokio::test]
async fn http_error_diagnostics_are_bounded_and_hide_worker_urls() {
    for failing_side in ["prefill", "decode"] {
        for large in [false, true] {
            let (pending, started, closed) = pending_peer().await;
            let failed = serve(Router::new().route(
                "/generate",
                post(move || {
                    let started = started.clone();
                    async move {
                        started.notified().await;
                        let mut text =
                            "Invalid parameter é at http://private-worker:30000/generate: detail "
                                .to_owned();
                        if large {
                            text.push_str(&"é".repeat(MAX_WORKER_ERROR_BYTES));
                        }
                        let chunks =
                            tokio_stream::once(Ok::<_, std::io::Error>(bytes::Bytes::from(text)))
                                .chain(futures_util::stream::pending());
                        (StatusCode::BAD_REQUEST, Body::from_stream(chunks))
                    }
                }),
            ))
            .await;
            let (p, d) = if failing_side == "prefill" {
                (&failed.url, &pending.url)
            } else {
                (&pending.url, &failed.url)
            };
            let response = tokio::time::timeout(Duration::from_secs(1), dispatch(p, d, true))
                .await
                .expect("a stalled 4xx body must not delay paired cancellation");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let payload: Value = serde_json::from_slice(&body).unwrap();
            let message = payload["error"]["message"].as_str().unwrap();
            assert!(message.contains("Invalid parameter é"), "{message}");
            assert!(message.contains("detail"), "{message}");
            assert!(!message.contains("private-worker"));
            assert!(!message.contains(&failed.url));
            assert!(message.len() <= MAX_WORKER_ERROR_BYTES + 100);
            tokio::time::timeout(Duration::from_secs(1), closed)
                .await
                .unwrap()
                .unwrap();
        }
    }
}

#[tokio::test]
async fn worker_5xx_does_not_expose_internal_urls_or_wait_for_body() {
    for failing_side in ["prefill", "decode"] {
        let (pending, started, closed) = pending_peer().await;
        let failed = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Body::from_stream(futures_util::stream::pending::<
                            Result<bytes::Bytes, std::io::Error>,
                        >()),
                    )
                }
            }),
        ))
        .await;
        let (p, d) = if failing_side == "prefill" {
            (&failed.url, &pending.url)
        } else {
            (&pending.url, &failed.url)
        };
        let response = dispatch(p, d, true).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(!String::from_utf8_lossy(&body).contains(&failed.url));
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn pending_decode_buffer_limit_cancels_pending_prefill() {
    const LIMIT: usize = 8192;
    let (prefill, started, closed) = pending_peer().await;
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                let event =
                    bytes::Bytes::from(format!("data: {{\"text\":\"{}\"}}\n\n", "x".repeat(128)));
                let chunks = tokio_stream::iter(
                    (0..128).map(move |_| Ok::<_, std::io::Error>(event.clone())),
                );
                (
                    [(CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(chunks),
                )
            }
        }),
    ))
    .await;
    let response = dispatch_with_limit(&prefill.url, &decode.url, true, LIMIT).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "pd_response_buffer_limit"
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn buffered_worker_body_limit_cancels_the_other_worker() {
    const LIMIT: usize = 8192;
    for failing_side in ["prefill", "decode"] {
        let (pending, started, closed) = pending_peer().await;
        let failed = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    Body::from(vec![b'x'; LIMIT + 1])
                }
            }),
        ))
        .await;
        let (p, d) = if failing_side == "prefill" {
            (&failed.url, &pending.url)
        } else {
            (&pending.url, &failed.url)
        };
        let response = dispatch_with_limit(p, d, false, LIMIT).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "pd_response_buffer_limit"
        );
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn oversized_sse_event_after_headers_returns_error_and_closes() {
    let mut router = create_test_pd_router();
    router.max_payload_size = 128;
    let chunks = tokio_stream::iter([
        Ok::<_, reqwest::Error>(bytes::Bytes::from_static(b"data: {\"text\":\"token\"}\n\n")),
        Ok(bytes::Bytes::from_static(b"data: {\"text\":\"unfinished")),
        Ok(bytes::Bytes::from(vec![b'x'; router.max_payload_size])),
    ]);
    let decode = worker("http://decode", WorkerType::Decode);
    let response = router.create_streaming_response(
        chunks,
        StatusCode::OK,
        None,
        false,
        None,
        worker(
            "http://prefill",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        decode.clone(),
        None,
    );
    assert_eq!(response.status(), StatusCode::OK);
    let mut events = response.into_body().into_data_stream().eventsource();
    let token = events.next().await.unwrap().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&token.data).unwrap()["text"],
        "token"
    );
    let failure = tokio::time::timeout(Duration::from_secs(1), events.next())
        .await
        .unwrap()
        .unwrap();
    assert!(
        failure.is_err(),
        "overflow must reach the SSE client as a transport error"
    );
    assert_eq!(decode.circuit_breaker().total_failures(), 1);
    assert_eq!(decode.circuit_breaker().total_successes(), 0);
}

#[tokio::test]
async fn sse_event_limit_is_independent_of_http_chunk_boundaries() {
    for newline in ["\n", "\r\n", "\r"] {
        let wire = format!(
            "\u{feff}: comment{newline}id: 7{newline}event: message{newline}\
             data: {{\"text\":{newline}data: \"data: [DONE] 你好\"}}{newline}{newline}\
             data: {{\"text\":\"second\"}}{newline}{newline}\
             data: [DONE]{newline}{newline}"
        );
        for chunk_size in [1, 11, wire.len()] {
            let mut router = create_test_pd_router();
            router.max_payload_size = 110;
            assert!(wire.len() > router.max_payload_size);
            let chunks: Vec<_> = wire
                .as_bytes()
                .chunks(chunk_size)
                .map(|part| Ok::<_, reqwest::Error>(bytes::Bytes::copy_from_slice(part)))
                .collect();
            let stream = tokio_stream::iter(chunks).chain(futures_util::stream::pending());
            let response = router.create_streaming_response(
                stream,
                StatusCode::OK,
                None,
                false,
                None,
                worker(
                    "http://prefill",
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ),
                worker("http://decode", WorkerType::Decode),
                None,
            );
            // eventsource-stream 0.2.3 slices a leading BOM at byte 1 and panics.
            // Check and strip only that BOM on the client side; keep all other boundaries.
            let mut bom = "\u{feff}".as_bytes();
            let client_stream = response.into_body().into_data_stream().map(move |chunk| {
                chunk.map(|chunk| {
                    let len = chunk.len().min(bom.len());
                    assert_eq!(&chunk[..len], &bom[..len]);
                    bom = &bom[len..];
                    chunk.slice(len..)
                })
            });
            let events = tokio::time::timeout(
                Duration::from_secs(1),
                client_stream.eventsource().collect::<Vec<_>>(),
            )
            .await
            .expect("a split terminal event must close the stream");
            let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
            assert_eq!(
                events.len(),
                3,
                "newline={newline:?}, chunk_size={chunk_size}"
            );
            assert_eq!(
                serde_json::from_str::<Value>(&events[0].data).unwrap()["text"],
                "data: [DONE] 你好"
            );
            assert_eq!(events[0].id, "7");
            assert_eq!(
                serde_json::from_str::<Value>(&events[1].data).unwrap()["text"],
                "second"
            );
            assert_eq!(events[2].data, "[DONE]");
        }
    }
}

#[test]
fn split_bom_preserves_the_first_sse_data_field() {
    for newline in ["\n", "\r\n", "\r"] {
        let payload = json!({"text": "你好"});
        let wire =
            format!("\u{feff}data: {payload}{newline}{newline}data: [DONE]{newline}{newline}");
        for chunk_size in [1, 2, wire.len()] {
            let mut buffer = SseEventBuffer::default();
            let mut events = Vec::new();
            for mut chunk in wire.as_bytes().chunks(chunk_size) {
                while let Some(event) = buffer.next_event(&mut chunk, wire.len()).unwrap() {
                    events.push(event);
                }
            }
            assert_eq!(
                events.len(),
                2,
                "newline={newline:?}, chunk_size={chunk_size}"
            );
            assert!(events[0].starts_with("\u{feff}data:".as_bytes()));
            assert_eq!(PDRouter::sse_payload(&events[0]).unwrap(), payload);
            assert_eq!(
                PDRouter::decode_event_ready(&events[0]).unwrap(),
                (true, false)
            );
            assert!(!PDRouter::check_prefill_event(&events[0]).unwrap());
            assert_eq!(PDRouter::sse_data(&events[1]), "[DONE]");
            assert_eq!(
                PDRouter::decode_event_ready(&events[1]).unwrap(),
                (true, true)
            );
            assert!(PDRouter::check_prefill_event(&events[1]).unwrap());
        }
    }
}

#[tokio::test]
async fn sse_event_limit_accumulates_across_chunks_and_accepts_exact_limit() {
    let event = "data: {\"text\":\"token\"}\r\n";
    let wire = format!("{event}\r\n");
    for limit in [event.len() - 1, event.len()] {
        for chunk_size in [1, wire.len()] {
            let mut router = create_test_pd_router();
            router.max_payload_size = limit;
            let chunks: Vec<_> = wire
                .as_bytes()
                .chunks(chunk_size)
                .map(|part| Ok::<_, reqwest::Error>(bytes::Bytes::copy_from_slice(part)))
                .collect();
            let response = router.create_streaming_response(
                tokio_stream::iter(chunks),
                StatusCode::OK,
                None,
                false,
                None,
                worker(
                    "http://prefill",
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ),
                worker("http://decode", WorkerType::Decode),
                None,
            );
            let mut events = response.into_body().into_data_stream().eventsource();
            let event = events.next().await.unwrap();
            assert_eq!(
                event.is_ok(),
                limit == wire.len() - 2,
                "limit={limit}, chunk_size={chunk_size}"
            );
        }
    }
}

#[tokio::test]
async fn ready_prefill_does_not_limit_the_size_of_a_decode_http_chunk() {
    let wire = format!(
        "{}data: [DONE]\n\n",
        "data: {\"text\":\"token\"}\n\n".repeat(20)
    );
    let expected = wire.clone();
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let wire = wire.clone();
            async move { ([(CONTENT_TYPE, "text/event-stream")], wire) }
        }),
    ))
    .await;
    let mut router = create_test_pd_router();
    router.max_payload_size = 64;
    let response = router
        .client
        .post(format!("{}/generate", decode.url))
        .send()
        .await
        .unwrap();
    let (ready, receiver) = oneshot::channel();
    ready.send(()).unwrap();
    let stream = router
        .prepare_decode_stream(response, receiver)
        .await
        .expect("ready prefill must bypass the pending-output buffer limit");
    let response = router.create_streaming_response(
        stream,
        StatusCode::OK,
        None,
        false,
        None,
        worker(
            "http://prefill",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ),
        worker(&decode.url, WorkerType::Decode),
        None,
    );
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body.as_ref(), expected.as_bytes());
}

#[tokio::test]
async fn prefill_sse_limits_events_unless_logprobs_require_retaining_the_body() {
    let wire = format!("{}data: [DONE]\n\n", "data: {\"text\":\"\"}\n\n".repeat(20));
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            let wire = wire.clone();
            async move { ([(CONTENT_TYPE, "text/event-stream")], wire) }
        }),
    ))
    .await;
    let mut router = create_test_pd_router();
    router.max_payload_size = 64;
    for return_logprob in [false, true] {
        let response = router
            .client
            .post(format!("{}/generate", prefill.url))
            .send()
            .await;
        let result = router
            .process_prefill_response(response, &prefill.url, return_logprob)
            .await;
        if return_logprob {
            assert_eq!(result.unwrap_err().status(), StatusCode::BAD_GATEWAY);
        } else {
            assert_eq!(result.unwrap().0, StatusCode::OK);
        }
    }
}

#[tokio::test]
async fn unterminated_sse_is_not_output_and_cancels_pending_prefill() {
    for wire in ["data: [DONE]", "data: {\"text\":\"token\"}\n"] {
        let (prefill, started, closed) = pending_peer().await;
        let decode = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    ([(CONTENT_TYPE, "text/event-stream")], wire)
                }
            }),
        ))
        .await;
        assert_eq!(
            dispatch(&prefill.url, &decode.url, true).await.status(),
            StatusCode::BAD_GATEWAY
        );
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn streaming_logprob_merge_preserves_sse_fields_and_chunk_independence() {
    let wire = "id: 9\r\nevent: message\r\ndata: {\"text\":\"[DONE] 你好\",\r\n\
                data: \"meta_info\":{\"input_token_logprobs\":[[0.2,2,\"d\"]]}}\r\n\r\n\
                data: [DONE]\r\n\r\n";
    for chunk_size in [1, wire.len()] {
        let router = create_test_pd_router();
        let chunks: Vec<_> = wire
            .as_bytes()
            .chunks(chunk_size)
            .map(|part| Ok::<_, reqwest::Error>(bytes::Bytes::copy_from_slice(part)))
            .collect();
        let response = router.create_streaming_response(
            tokio_stream::iter(chunks),
            StatusCode::OK,
            Some(json!([[0.1, 1, "p"]])),
            true,
            None,
            worker(
                "http://prefill",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            worker("http://decode", WorkerType::Decode),
            None,
        );
        let events: Vec<_> = response
            .into_body()
            .into_data_stream()
            .eventsource()
            .collect()
            .await;
        let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event, "message");
        assert_eq!(events[0].id, "9");
        let payload: Value = serde_json::from_str(&events[0].data).unwrap();
        assert_eq!(payload["text"], "[DONE] 你好");
        assert_eq!(
            payload["meta_info"]["input_token_logprobs"],
            json!([[0.1, 1, "p"], [0.2, 2, "d"]])
        );
        assert!(payload.get("type").is_none());
        assert_eq!(events[1].data, "[DONE]");
    }
}

#[tokio::test]
async fn default_limit_accepts_logprob_bodies_larger_than_eight_mib() {
    let payload = Json(json!({
        "text": "token", "padding": "x".repeat(8 * 1024 * 1024),
        "meta_info": {"input_token_logprobs": [[0.1, 1, "p"]]}
    }));
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            let payload = payload.clone();
            async move { payload }
        }),
    ))
    .await;
    let router = create_test_pd_router();
    let mut ctx = context(false);
    ctx.return_logprob = true;
    let response = router
        .execute_dual_dispatch_internal(
            None,
            json!({"text":"prompt", "return_logprob":true}),
            ctx,
            worker(
                &prefill.url,
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            worker(&prefill.url, WorkerType::Decode),
            tokio::time::Instant::now() + Duration::from_secs(10),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let payload: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        payload["meta_info"]["input_token_logprobs"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn nonstream_decode_json_error_cancels_pending_prefill() {
    let (prefill, started, closed) = pending_peer().await;
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                Json(json!({"error":{"message":"decode failed", "code":500}}))
            }
        }),
    ))
    .await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, false).await.status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[test]
fn json_error_detection_parses_each_body_once() {
    // Every fragment ends in a nested closing brace: the former suffix heuristic
    // reparsed the growing logprob prefix for every one of these fragments.
    let item = json!({"token": "escaped \\\" quote, slash \\, braces } ] { [ 你好"});
    for side in ["prefill", "decode"] {
        for split_at_braces in [false, true] {
            let wire = serde_json::to_vec(&json!({"logprobs": vec![item.clone(); 1024]})).unwrap();
            let mut detector = JsonErrorDetector::default();
            let mut body = Vec::new();
            let chunks: Vec<_> = if split_at_braces {
                wire.split_inclusive(|byte| matches!(byte, b'}' | b']'))
                    .collect()
            } else {
                wire.chunks(1).collect()
            };
            for chunk in chunks {
                body.extend_from_slice(chunk);
                detector.check(chunk, &body, side).unwrap();
                assert_eq!(
                    detector.parse_attempts,
                    usize::from(body.len() == wire.len())
                );
            }
            for chunk in [b" ".as_slice(), b"\n\t", b" "] {
                body.extend_from_slice(chunk);
                detector.check(chunk, &body, side).unwrap();
                assert_eq!(detector.parse_attempts, 1);
            }
        }
    }
}

#[test]
fn json_error_detection_preserves_nested_errors_and_invalid_json_fallback() {
    for (wire, status) in [
        (
            r#"[{"response":{"error":{"code":503,"message":"failed"}}}]"#,
            Some(StatusCode::SERVICE_UNAVAILABLE),
        ),
        (
            r#"{"meta_info":{"finish_reason":{"type":"abort"}}}"#,
            Some(StatusCode::from_u16(499).unwrap()),
        ),
        (r#"{"broken":]}"#, None),
        (r#"{"incomplete":[{}"#, None),
    ] {
        let mut detector = JsonErrorDetector::default();
        let mut body = Vec::new();
        let mut error_status = None;
        for chunk in wire.as_bytes().chunks(1) {
            body.extend_from_slice(chunk);
            if let Err(response) = detector.check(chunk, &body, "decode") {
                error_status = Some(response.status());
            }
        }
        assert_eq!(error_status, status, "{wire}");
        assert!(detector.parse_attempts <= 1);
    }
}

#[tokio::test]
async fn prefill_http_failure_cancels_pending_decode_without_reading_error_body() {
    let (decode, started, closed) = pending_peer().await;
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                (
                    StatusCode::BAD_REQUEST,
                    Body::from_stream(futures_util::stream::pending::<
                        Result<bytes::Bytes, std::io::Error>,
                    >()),
                )
            }
        }),
    ))
    .await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::BAD_REQUEST
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn prefill_json_errors_and_native_abort_cancel_decode() {
    for (payload, expected_status) in [
        (
            json!({"error":{"message":"KV load failed", "code":500}}),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!([{"meta_info":{"finish_reason":{"type":"abort", "message":"bootstrap failed", "status_code":500}}}]),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"meta_info":{"finish_reason":{"type":"abort", "message":"user cancelled"}}}),
            StatusCode::from_u16(499).unwrap(),
        ),
    ] {
        let (decode, started, closed) = pending_peer().await;
        let prefill = serve(Router::new().route(
            "/generate",
            post(move || {
                let (started, payload) = (started.clone(), payload.clone());
                async move {
                    started.notified().await;
                    Json(payload)
                }
            }),
        ))
        .await;
        assert_eq!(
            dispatch(&prefill.url, &decode.url, true).await.status(),
            expected_status
        );
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn complete_json_error_is_detected_without_waiting_for_eof() {
    let (decode, started, closed) = pending_peer().await;
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                let chunks =
                    tokio_stream::once(Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                        b"{\"error\":{\"message\":\"load failed\",\"code\":500}}",
                    )))
                    .chain(futures_util::stream::pending());
                (
                    [(CONTENT_TYPE, "application/json")],
                    Body::from_stream(chunks),
                )
            }
        }),
    ))
    .await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn decode_json_error_cancels_prefill_without_waiting_for_body_end() {
    for stream in [false, true] {
        let (prefill, started, closed) = pending_peer().await;
        let decode = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    let chunks = tokio_stream::iter([
                        Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                            b"{\"error\":{\"message\":\"KV handshake",
                        )),
                        Ok(bytes::Bytes::from_static(b" failed\",\"code\":503}}")),
                    ])
                    .chain(futures_util::stream::pending());
                    (
                        [(CONTENT_TYPE, "application/json")],
                        Body::from_stream(chunks),
                    )
                }
            }),
        ))
        .await;
        let (response, metrics) =
            observe_metrics(dispatch(&prefill.url, &decode.url, stream)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let worker_metrics: Vec<_> = metrics
            .lines()
            .filter(|line| line.starts_with("smg_worker_errors_total{"))
            .collect();
        assert_eq!(worker_metrics.len(), 1, "{metrics}");
        assert!(worker_metrics[0].contains("worker_type=\"decode\""));
        assert!(worker_metrics[0].ends_with(" 1"));
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn fragmented_prefill_sse_error_cancels_decode_before_eof() {
    let (decode, started, closed) = pending_peer().await;
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                let chunks = tokio_stream::iter([
                    Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                        b": keepalive\r\n\r\ndata: {\"err",
                    )),
                    Ok(bytes::Bytes::from_static(
                        b"or\":{\"message\":\"KV failed\",\"code\":500}}\r\n\r\n",
                    )),
                ])
                .chain(futures_util::stream::pending());
                (
                    [(CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(chunks),
                )
            }
        }),
    ))
    .await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn decode_error_after_role_or_token_cancels_prefill_and_records_metrics_once() {
    for first_event in [
        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".as_slice(),
        b"data: {\"text\":\"first token\"}\n\n".as_slice(),
        b"event: response.created\ndata: {\"response\":{\"status\":\"in_progress\"}}\n\n"
            .as_slice(),
    ] {
        let (prefill, started, closed) = pending_peer().await;
        let decode = serve(Router::new().route(
            "/generate",
            post(move || {
                let started = started.clone();
                async move {
                    started.notified().await;
                    let chunks = tokio_stream::iter([
                        Ok::<_, std::io::Error>(bytes::Bytes::from_static(first_event)),
                        Ok(bytes::Bytes::from_static(
                            b"data: {\"error\":{\"message\":\"KV handshake failed\",\"code\":503}}\n\n",
                        )),
                    ])
                    .chain(futures_util::stream::pending());
                    (
                        [(CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(chunks),
                    )
                }
            }),
        ))
        .await;
        let (response, metrics) = observe_metrics(dispatch(&prefill.url, &decode.url, true)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("KV handshake failed"));
        let stream_metrics: Vec<_> = metrics
            .lines()
            .filter(|line| line.starts_with("smg_router_stream_outcome_total{"))
            .collect();
        assert_eq!(stream_metrics.len(), 1, "{metrics}");
        assert!(stream_metrics[0].contains("outcome=\"stream_error_event\""));
        assert!(stream_metrics[0].contains(&format!("worker_url=\"{}\"", decode.url)));
        assert!(stream_metrics[0].ends_with(" 1"));
        let worker_metrics: Vec<_> = metrics
            .lines()
            .filter(|line| line.starts_with("smg_worker_errors_total{"))
            .collect();
        assert_eq!(worker_metrics.len(), 1, "{metrics}");
        assert!(worker_metrics[0].contains("worker_type=\"decode\""));
        assert!(worker_metrics[0].ends_with(" 1"));
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn decode_disconnect_after_token_does_not_wait_for_prefill() {
    let (prefill, started, closed) = pending_peer().await;
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                let chunks = tokio_stream::once(Ok(bytes::Bytes::from_static(
                    b"data: {\"text\":\"token\"}\n\n",
                )))
                .chain(futures_util::stream::once(async {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Err(std::io::Error::other("decode connection reset"))
                }));
                (
                    [(CONTENT_TYPE, "text/event-stream")],
                    Body::from_stream(chunks),
                )
            }
        }),
    ))
    .await;
    let (response, metrics) = observe_metrics(dispatch(&prefill.url, &decode.url, true)).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let lines: Vec<_> = metrics
        .lines()
        .filter(|line| line.starts_with("smg_router_stream_outcome_total{"))
        .collect();
    assert_eq!(lines.len(), 1, "{metrics}");
    assert!(lines[0].contains("outcome=\"upstream_error\""));
    assert!(lines[0].ends_with(" 1"));
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn selection_failure_does_not_record_worker_errors() {
    let router = create_test_pd_router();
    let (response, metrics) = observe_metrics(router.execute_dual_dispatch(
        None,
        &json!({"text":"prompt", "stream":true}),
        context(true),
    ))
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(!metrics.contains("smg_worker_errors_total{"), "{metrics}");
    assert!(
        !metrics.contains("smg_router_stream_outcome_total{"),
        "{metrics}"
    );
    assert!(
        metrics.contains("smg_router_request_errors_total{"),
        "{metrics}"
    );
}

#[test]
fn streaming_error_closes_response_and_records_metrics_once() {
    for (error_event, worker_failed) in [
        ("data: {\"error\":{\"message\":\"KV handshake failed\",\"code\":503}}\n\n", true),
        ("data: {\"text\":\"\",\"meta_info\":{\"finish_reason\":{\"type\":\"abort\",\"message\":\"KV handshake failed\",\"status_code\":503}}}\n\n", true),
        ("data: {\"error\":{\"message\":\"invalid request\",\"code\":400}}\n\n", false),
        ("data: {\"text\":\"\",\"meta_info\":{\"finish_reason\":{\"type\":\"abort\",\"message\":\"user abort\"}}}\n\n", false),
        ("event: response.failed\ndata: {\"response\":{\"status\":\"failed\",\"error\":{\"code\":503,\"message\":\"generation failed\"}}}\n\n", true),
        ("data: {\"type\":\"response.error\",\"error\":{\"code\":400,\"message\":\"invalid request\"}}\n\n", false),
        ("event: error\ndata: {\"code\":503,\"message\":\"generation failed\"}\n\n", true),
    ] {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        // Keep the recorder active for the spawned forwarding task as well.
        metrics::with_local_recorder(&recorder, || runtime.block_on(async {
            let router = create_test_pd_router();
            let token = b"data: {\"text\":\"token\"}\n\n";
            let split = error_event.len() / 2;
            let chunks = [
                bytes::Bytes::from_static(token),
                bytes::Bytes::copy_from_slice(&error_event.as_bytes()[..split]),
                bytes::Bytes::copy_from_slice(&error_event.as_bytes()[split..]),
            ];
            let stream = tokio_stream::iter(chunks.into_iter().map(Ok::<_, reqwest::Error>))
                .chain(futures_util::stream::pending());
            let prefill = worker("http://prefill", WorkerType::Prefill { bootstrap_port: None });
            let decode = worker("http://decode", WorkerType::Decode);
            let response = router.create_streaming_response(
                stream, StatusCode::OK, None, false, None,
                prefill.clone(), decode.clone(), Some("test-model"),
            );
            let body = tokio::time::timeout(Duration::from_secs(1),
                axum::body::to_bytes(response.into_body(), usize::MAX))
                .await.expect("an error event must close the stream even if decode stalls").unwrap();
            assert_eq!(body.as_ref(), [token.as_slice(), error_event.as_bytes()].concat());
            let metrics = handle.render();
            let stream_metrics: Vec<_> = metrics.lines().filter(|line| line.starts_with("smg_router_stream_outcome_total{")).collect();
            assert_eq!(stream_metrics.len(), 1, "{metrics}");
            assert!(stream_metrics[0].contains("outcome=\"stream_error_event\""));
            assert!(stream_metrics[0].ends_with(" 1"));
            assert_eq!(prefill.circuit_breaker().total_successes(), 0);
            assert_eq!(prefill.circuit_breaker().total_failures(), 0);
            assert_eq!(decode.circuit_breaker().total_successes(), u64::from(!worker_failed));
            assert_eq!(decode.circuit_breaker().total_failures(), u64::from(worker_failed));
            let worker_metrics: Vec<_> = metrics.lines().filter(|line| line.starts_with("smg_worker_errors_total{")).collect();
            assert_eq!(worker_metrics.len(), usize::from(worker_failed), "{metrics}");
            if worker_failed {
                assert!(worker_metrics[0].contains("worker_type=\"decode\""));
                assert!(worker_metrics[0].ends_with(" 1"));
            }
        }));
    }
}

#[tokio::test]
async fn responses_terminal_events_finish_both_workers_without_eof() {
    for event in [
        "event: response.completed\ndata: {\"response\":{\"status\":\"completed\",\"error\":null}}\n\n",
        "event: response.incomplete\ndata: {\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
    ] {
        let app = Router::new().route(
            "/generate",
            post(move || async move {
                let stream = tokio_stream::once(Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                    event.as_bytes(),
                )))
                .chain(futures_util::stream::pending());
                ([(CONTENT_TYPE, "text/event-stream")], Body::from_stream(stream))
            }),
        );
        let prefill = serve(app.clone()).await;
        let decode = serve(app).await;
        let response = dispatch(&prefill.url, &decode.url, true).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = tokio::time::timeout(
            Duration::from_secs(1),
            axum::body::to_bytes(response.into_body(), usize::MAX),
        )
        .await
        .expect("Responses terminal events must finish without EOF or [DONE]")
        .unwrap();
        assert_eq!(body.as_ref(), event.as_bytes());
    }
}

#[tokio::test]
async fn early_terminated_sse_has_valid_http_framing() {
    const TOKEN: &[u8] = b"data: {\"text\":\"token\"}\n\n";
    const TRAILER: &[u8] = b"data: [DONE]\n\n";
    for terminal in [
        b"data: {\"error\":{\"code\":503,\"message\":\"generation failed\"}}\n\n".as_slice(),
        b"event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n"
            .as_slice(),
        b"event: response.incomplete\ndata: {\"response\":{\"status\":\"incomplete\"}}\n\n"
            .as_slice(),
    ] {
        // Client/TLS setup belongs before the timed HTTP request, as in serving.
        let router = Arc::new(create_test_pd_router());
        let gateway = serve(Router::new().route(
            "/generate",
            post(move || {
                let router = router.clone();
                async move {
                    // Separate input chunks keep the suffix out of the terminal chunk.
                    let chunks = [TOKEN, terminal, TRAILER].map(bytes::Bytes::from_static);
                    let stream =
                        tokio_stream::iter(chunks.into_iter().map(Ok::<_, reqwest::Error>));
                    let mut headers = HeaderMap::new();
                    headers.insert(
                        CONTENT_LENGTH,
                        HeaderValue::from_str(
                            &(TOKEN.len() + terminal.len() + TRAILER.len()).to_string(),
                        )
                        .unwrap(),
                    );
                    headers.insert("x-request-id", HeaderValue::from_static("framing-test"));
                    router.create_streaming_response(
                        stream,
                        StatusCode::OK,
                        None,
                        false,
                        Some(headers),
                        worker(
                            "http://prefill",
                            WorkerType::Prefill {
                                bootstrap_port: None,
                            },
                        ),
                        worker("http://decode", WorkerType::Decode),
                        None,
                    )
                }
            }),
        ))
        .await;
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        // Use real HTTP: collecting an axum Body alone cannot detect a stale length.
        let response = client
            .post(format!("{}/generate", gateway.url))
            .send()
            .await
            .expect("gateway must send a valid HTTP response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
        assert!(!response.headers().contains_key(CONTENT_LENGTH));
        assert_eq!(response.headers()["x-request-id"], "framing-test");
        let body = response
            .bytes()
            .await
            .expect("early SSE termination must not cause a Content-Length mismatch");
        assert_eq!(body.as_ref(), [TOKEN, terminal].concat());
    }
}

#[tokio::test]
async fn terminal_sse_discards_coalesced_and_split_trailing_junk() {
    for terminal in [
        "data: [DONE]\n\n",
        "event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n",
        "event: response.incomplete\ndata: {\"response\":{\"status\":\"incomplete\"}}\n\n",
        "data: {\"error\":{\"code\":503}}\n\n",
        "data: {\"meta_info\":{\"finish_reason\":{\"type\":\"abort\"}}}\n\n",
        "event: response.failed\ndata: {\"response\":{\"status\":\"failed\"}}\n\n",
        "event: error\ndata: {\"code\":400}\n\n",
    ] {
        for newline in ["\n", "\r\n", "\r"] {
            let token = "data: {\"text\":\"token\",\"meta_info\":{\"input_token_logprobs\":[[0.2,2,\"d\"]]}}\n\n"
                .replace('\n', newline);
            let terminal = terminal.replace('\n', newline);
            let wire = format!("{token}{terminal}data: trailing-junk{newline}{newline}");
            for chunk_size in [1, 2, wire.len()] {
                for (return_logprob, prefill_logprobs) in [
                    (false, None),
                    (true, None),
                    (true, Some(json!([[0.1, 1, "p"]]))),
                    (true, Some(Value::Null)),
                ] {
                    let merges = prefill_logprobs.as_ref().is_some_and(Value::is_array);
                    let chunks: Vec<_> = wire
                        .as_bytes()
                        .chunks(chunk_size)
                        .map(|part| Ok::<_, reqwest::Error>(bytes::Bytes::copy_from_slice(part)))
                        .collect();
                    let response = create_test_pd_router().create_streaming_response(
                        tokio_stream::iter(chunks).chain(futures_util::stream::pending()),
                        StatusCode::OK,
                        prefill_logprobs,
                        return_logprob,
                        None,
                        worker(
                            "http://prefill",
                            WorkerType::Prefill {
                                bootstrap_port: None,
                            },
                        ),
                        worker("http://decode", WorkerType::Decode),
                        None,
                    );
                    let body = tokio::time::timeout(
                        Duration::from_secs(1),
                        axum::body::to_bytes(response.into_body(), usize::MAX),
                    )
                    .await
                    .expect("terminal must close without EOF")
                    .unwrap();
                    assert!(!String::from_utf8_lossy(&body).contains("trailing-junk"));
                    if !return_logprob {
                        let mut expected = format!("{token}{terminal}");
                        if expected.ends_with('\r') {
                            expected.push('\n');
                        }
                        assert_eq!(body.as_ref(), expected.as_bytes());
                    }
                    let events: Vec<_> = tokio_stream::once(Ok::<_, std::io::Error>(body))
                        .eventsource()
                        .collect()
                        .await;
                    let events: Vec<_> = events.into_iter().map(Result::unwrap).collect();
                    assert_eq!(events.len(), 2, "newline={newline:?}, chunk={chunk_size}");
                    let token: Value = serde_json::from_str(&events[0].data).unwrap();
                    assert_eq!(
                        token["meta_info"]["input_token_logprobs"]
                            .as_array()
                            .unwrap()
                            .len(),
                        if merges { 2 } else { 1 }
                    );
                    assert_eq!(
                        events[1].data,
                        PDRouter::sse_data(terminal.as_bytes()).trim_end()
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn terminal_decode_before_prefill_ready_ignores_trailing_junk() {
    for terminal in [
        "data: [DONE]\n\n",
        "event: response.completed\ndata: {\"response\":{\"status\":\"completed\"}}\n\n",
    ] {
        let wire = format!("{terminal}data: invalid-json\n\n");
        let decode = serve(Router::new().route(
            "/generate",
            post(move || {
                let wire = wire.clone();
                async move {
                    let chunks =
                        tokio_stream::once(Ok::<_, std::io::Error>(bytes::Bytes::from(wire)))
                            .chain(futures_util::stream::pending());
                    (
                        [(CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(chunks),
                    )
                }
            }),
        ))
        .await;
        let router = create_test_pd_router();
        let response = router
            .client
            .post(format!("{}/generate", decode.url))
            .send()
            .await
            .unwrap();
        let (_ready, receiver) = oneshot::channel();
        let stream = tokio::time::timeout(
            Duration::from_secs(1),
            router.prepare_decode_stream(response, receiver),
        )
        .await
        .unwrap()
        .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let body: Vec<_> = chunks
            .into_iter()
            .flat_map(|chunk| chunk.unwrap().to_vec())
            .collect();
        assert_eq!(body, terminal.as_bytes());
    }
}

#[test]
fn error_after_terminal_does_not_change_stream_outcome() {
    let wire = b"data: [DONE]\n\ndata: {\"error\":{\"code\":503}}\n\n";
    for chunk_size in [1, wire.len()] {
        let decode = worker("http://decode", WorkerType::Decode);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let body = metrics::with_local_recorder(&recorder, || {
            runtime.block_on(async {
                let chunks: Vec<_> = wire
                    .chunks(chunk_size)
                    .map(|chunk| Ok::<_, reqwest::Error>(bytes::Bytes::copy_from_slice(chunk)))
                    .collect();
                let response = create_test_pd_router().create_streaming_response(
                    tokio_stream::iter(chunks),
                    StatusCode::OK,
                    None,
                    false,
                    None,
                    worker(
                        "http://prefill",
                        WorkerType::Prefill {
                            bootstrap_port: None,
                        },
                    ),
                    decode.clone(),
                    None,
                );
                axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap()
            })
        });
        let metrics = handle.render();
        assert_eq!(body.as_ref(), b"data: [DONE]\n\n");
        let outcomes: Vec<_> = metrics
            .lines()
            .filter(|line| line.starts_with("smg_router_stream_outcome_total{"))
            .collect();
        assert_eq!(outcomes.len(), 1, "{metrics}");
        assert!(outcomes[0].contains("outcome=\"ok\""), "{metrics}");
        assert_eq!(decode.circuit_breaker().total_failures(), 0);
        assert_eq!(decode.circuit_breaker().total_successes(), 1);
    }
}

async fn broken_body() -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 8192];
        assert!(socket.read(&mut buf).await.unwrap() > 0);
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
            .await
            .unwrap();
    });
    TestServer { url, task }
}

#[tokio::test]
async fn prefill_body_read_failure_is_terminal() {
    let prefill = broken_body().await;
    let (decode, _, _) = pending_peer().await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::BAD_GATEWAY
    );
}

#[tokio::test]
async fn nonstream_decode_body_read_failure_does_not_wait_for_prefill() {
    for stream in [false, true] {
        let (prefill, _, _) = pending_peer().await;
        let decode = broken_body().await;
        assert_eq!(
            dispatch(&prefill.url, &decode.url, stream).await.status(),
            StatusCode::BAD_GATEWAY
        );
    }
}

#[tokio::test]
async fn early_decode_success_still_waits_for_prefill_validation() {
    let decode =
        serve(Router::new().route("/generate", post(|| async { Json(json!({"text":"ok"})) })))
            .await;
    let prefill = serve(Router::new().route(
        "/generate",
        post(|| async {
            tokio::time::sleep(Duration::from_millis(30)).await;
            Json(json!({"error":"external KV load failed"}))
        }),
    ))
    .await;
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::BAD_GATEWAY
    );
}

fn registered_router(p: &str, d: &str) -> PDRouter {
    let router = create_test_pd_router();
    router.worker_registry.register(worker(
        p,
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ));
    router
        .worker_registry
        .register(worker(d, WorkerType::Decode));
    router
}

#[tokio::test]
async fn failed_pd_request_returns_without_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let prefill = serve(Router::new().route(
        "/generate",
        post(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            async { StatusCode::SERVICE_UNAVAILABLE }
        }),
    ))
    .await;
    let decode =
        serve(Router::new().route("/generate", post(|| async { Json(json!({"text":"ok"})) })))
            .await;
    let mut router = registered_router(&prefill.url, &decode.url);
    router.request_timeout = Duration::from_millis(150);
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        router.execute_dual_dispatch(None, &json!({"text":"prompt"}), context(false)),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn deadline_cancels_both_silent_workers() {
    let (prefill, _, p_closed) = pending_peer().await;
    let (decode, _, d_closed) = pending_peer().await;
    let mut router = registered_router(&prefill.url, &decode.url);
    router.request_timeout = Duration::from_millis(150);
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        router.execute_dual_dispatch(None, &json!({"text":"prompt"}), context(true)),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    for closed in [p_closed, d_closed] {
        tokio::time::timeout(Duration::from_secs(1), closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn failed_pd_request_is_not_replayed_on_another_pair() {
    let rooms = Arc::new(std::sync::Mutex::new(Vec::new()));
    let requests = rooms.clone();
    let prefill_app = Router::new().route(
        "/generate",
        post(move |Json(payload): Json<Value>| {
            let attempt = {
                let mut rooms = requests.lock().unwrap();
                rooms.push(payload["bootstrap_room"].clone());
                rooms.len()
            };
            async move {
                if attempt == 1 {
                    StatusCode::SERVICE_UNAVAILABLE.into_response()
                } else {
                    Json(json!({"text":""})).into_response()
                }
            }
        }),
    );
    let failed = serve(prefill_app.clone()).await;
    let healthy = serve(prefill_app).await;
    let decode =
        serve(Router::new().route("/generate", post(|| async { Json(json!({"text":"ok"})) })))
            .await;
    let router = registered_router(&failed.url, &decode.url);
    router.worker_registry.register(worker(
        &healthy.url,
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ));
    let response = router
        .execute_dual_dispatch(None, &json!({"text":"prompt"}), context(false))
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let rooms = rooms.lock().unwrap();
    assert_eq!(rooms.len(), 1);
}

#[tokio::test]
async fn cancelled_peer_is_not_penalized_by_the_circuit_breaker() {
    use crate::core::circuit_breaker::CircuitState;
    let (prefill, started, closed) = pending_peer().await;
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                StatusCode::SERVICE_UNAVAILABLE
            }
        }),
    ))
    .await;
    let router = create_test_pd_router();
    let p = worker(
        &prefill.url,
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    );
    let d = worker(&decode.url, WorkerType::Decode);
    let response = router
        .execute_dual_dispatch_internal(
            None,
            json!({"text":"prompt"}),
            context(false),
            p.clone(),
            d.clone(),
            tokio::time::Instant::now() + Duration::from_secs(2),
        )
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(p.circuit_breaker().state(), CircuitState::Closed);
    assert_eq!(d.circuit_breaker().state(), CircuitState::Open);
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn successful_stream_is_forwarded_once_without_replay() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let prefill =
        serve(Router::new().route("/generate", post(|| async { Json(json!({"text":""})) }))).await;
    const SSE: &str = "data: {\"text\":\"token\"}\n\ndata: [DONE]\n\n";
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            async { ([(CONTENT_TYPE, "text/event-stream")], SSE) }
        }),
    ))
    .await;
    let router = registered_router(&prefill.url, &decode.url);
    let response = router
        .execute_dual_dispatch(
            None,
            &json!({"text":"prompt", "stream":true}),
            context(true),
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body.as_ref(), SSE.as_bytes());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn healthy_logprob_responses_remain_compatible() {
    for stream in [false, true] {
        let prefill = serve(Router::new().route(
            "/generate",
            post(|| async { Json(json!({"meta_info":{"input_token_logprobs":[[0.1,1,"p"]]}})) }),
        ))
        .await;
        let decode = serve(Router::new().route("/generate", post(move || async move {
            if stream {
                let chunks = tokio_stream::iter([
                    "data: {\"text\":\"\",\"meta_info\":{\"input_token_logprobs\":[[0.2,2,\"d\"]]}}\n\n",
                    "data: {\"text\":\"token\",\"meta_info\":{\"input_token_logprobs\":[[0.2,2,\"d\"]]}}\n\n",
                    "data: [DONE]\n\n",
                ]).then(|chunk| async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok::<_, std::io::Error>(bytes::Bytes::from_static(chunk.as_bytes()))
                });
                ([(CONTENT_TYPE, "text/event-stream")], Body::from_stream(chunks)).into_response()
            } else {
                Json(json!({"text":"token", "meta_info":{"input_token_logprobs":[[0.2,2,"d"]]}})).into_response()
            }
        }))).await;
        let router = create_test_pd_router();
        let mut ctx = context(stream);
        ctx.return_logprob = true;
        let response = router
            .execute_dual_dispatch_internal(
                None,
                json!({"text":"prompt", "stream":stream}),
                ctx,
                worker(
                    &prefill.url,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                ),
                worker(&decode.url, WorkerType::Decode),
                tokio::time::Instant::now() + Duration::from_secs(2),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        if !stream {
            assert!(!response.headers().contains_key(CONTENT_LENGTH));
        }
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: Value = if stream {
            let text = std::str::from_utf8(&body).unwrap();
            let token_event = text.lines().find(|line| line.contains("token")).unwrap();
            serde_json::from_str(token_event.strip_prefix("data: ").unwrap()).unwrap()
        } else {
            serde_json::from_slice(&body).unwrap()
        };
        assert_eq!(
            payload["meta_info"]["input_token_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
}
