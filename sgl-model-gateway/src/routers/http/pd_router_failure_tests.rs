use std::sync::atomic::{AtomicUsize, Ordering};

use axum::{routing::post, Json, Router};
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
    let router = create_test_pd_router();
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
    assert_eq!(
        dispatch(&prefill.url, &absent, true).await.status(),
        StatusCode::BAD_GATEWAY
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
async fn decode_error_after_role_event_cancels_prefill_before_first_token() {
    let (prefill, started, closed) = pending_peer().await;
    let decode = serve(Router::new().route(
        "/generate",
        post(move || {
            let started = started.clone();
            async move {
                started.notified().await;
                let chunks = tokio_stream::iter([
                    Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                        b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
                    )),
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
    assert_eq!(
        dispatch(&prefill.url, &decode.url, true).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    tokio::time::timeout(Duration::from_secs(1), closed)
        .await
        .unwrap()
        .unwrap();
}

async fn broken_body() -> TestServer {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 8192];
        socket.read(&mut buf).await.unwrap();
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
async fn deadline_includes_retry_backoff() {
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
    router.retry_config.initial_backoff_ms = 2000;
    router.retry_config.max_backoff_ms = 2000;
    router.retry_config.jitter_factor = 0.0;
    let response = tokio::time::timeout(
        Duration::from_secs(2),
        router.execute_dual_dispatch(None, &json!({"text":"prompt"}), context(false)),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
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
async fn retry_selects_a_healthy_pair_with_a_new_bootstrap_room() {
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
    let mut router = registered_router(&failed.url, &decode.url);
    router.worker_registry.register(worker(
        &healthy.url,
        WorkerType::Prefill {
            bootstrap_port: None,
        },
    ));
    router.retry_config.max_retries = 2;
    router.retry_config.initial_backoff_ms = 1;
    router.retry_config.max_backoff_ms = 1;
    router.retry_config.jitter_factor = 0.0;
    let response = router
        .execute_dual_dispatch(None, &json!({"text":"prompt"}), context(false))
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let rooms = rooms.lock().unwrap();
    assert_eq!(rooms.len(), 2);
    assert_ne!(rooms[0], rooms[1]);
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
    const SSE: &str =
        "data: {\"text\":\"token\"}\n\ndata: {\"error\":\"later failure\"}\n\ndata: [DONE]\n\n";
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
