# SGLang Model Gateway Python Bindings

This directory contains the Python bindings for the SGLang Router, built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Directory Structure

```
bindings/python/
├── src/                    # Source code (src layout)
│   ├── lib.rs              # Rust/PyO3 bindings implementation
│   └── sglang_router/      # Python source code
│       ├── __init__.py
│       ├── version.py
│       ├── launch_server.py
│       ├── launch_router.py
│       ├── router.py
│       ├── router_args.py
│       └── mini_lb.py
├── tests/                  # Python unit tests
│   ├── conftest.py
│   ├── test_validation.py
│   ├── test_arg_parser.py
│   ├── test_router_config.py
│   └── test_startup_sequence.py
├── Cargo.toml              # Rust package configuration for bindings
├── pyproject.toml          # Python package configuration
├── setup.py                # Setup configuration
├── MANIFEST.in             # Package manifest
├── .coveragerc             # Test coverage configuration
└── README.md               # This file
```

## Building

### Development Build

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd sgl-model-gateway/bindings/python
maturin develop --features vendored-openssl
```

### Production Build

```bash
# Build wheel
cd sgl-model-gateway/bindings/python
maturin build --release --out dist --features vendored-openssl

# Install the built wheel
pip install dist/sglang_router-*.whl
```

## HTTP streaming outcome metrics

The native `sglang-router` wheel exports
`smg_router_stream_outcome_total{router_type,backend_type,model,worker_url,outcome}`
on its configured Prometheus endpoint. This counter observes HTTP regular and
HTTP PD streams whose upstream response status is 2xx; PD labels identify the
decode worker. Requests without an explicit model use the worker's model ID.

| `outcome` | Meaning |
| --- | --- |
| `ok` | Upstream reached EOF, or the existing PD loop recognized `[DONE]`. |
| `stream_error_event` | An engine SSE line starts with `data:` and a JSON object whose first key is `error`, or PD detects an error/abort payload before forwarding. |
| `upstream_error` | Upstream transport/read failure, or an invalid/empty PD stream before forwarding; transport errors take precedence over in-band errors. |
| `client_disconnect` | The response stream was dropped before upstream completion or error. |

PD requests are sent once. Either worker's failure cancels its paired request
and returns an error without retrying. Decode stays monitored while Prefill is
pending, including after the first token. A forwarded error event closes the
PD stream after that event is sent. Decode errors detected after upstream
2xx headers but before forwarding are also counted in the stream metric.
Prefill failures and non-2xx responses remain in
`smg_http_responses_total` and `smg_router_request_errors_total`. The stream counter does not
cover gRPC or external OpenAI response-processing paths. `ok` is an upstream
completion observation, not proof that the client received every byte. A PD
failure is attributed to the decode stream and does not identify the failing
prefill as its root cause. No `expired` or `aborted` classifications are added.
Series appear when an observed stream ends or an early decode error is detected.
Error events are logged with their worker URL; each logged SSE line is capped
at 4 KiB. Worker error counters only record the side that actually failed.

```promql
sum by (outcome) (rate(smg_router_stream_outcome_total[5m]))
```

Build and install this directory's wheel to use the metric with the existing
`python -m sglang_router.launch_router` entry point. Updating only the `sglang`
Engine wheel or building `experimental/sgl-router` does not update this package.

## Testing

```bash
# Run Python unit tests (after maturin develop)
cd sgl-model-gateway/bindings/python
pytest tests/
```

## Configuration

- **pyproject.toml**: Defines package metadata, dependencies, and build configuration
- **python-source**: Set to `"src"` indicating Python source uses the src layout
- **module-name**: `sglang_router.sglang_router_rs` - the Rust extension module name

## Notes

- The Rust bindings source code is located in `src/lib.rs`
- The bindings have their own `Cargo.toml` in this directory
- The main sglang-router library is located in `../../` and is used as a dependency
- The package includes both Python code and Rust extensions built with PyO3
- PyO3 types are prefixed with `Py` in Rust but exposed to Python without the prefix using the `name` attribute
