# chaos-mesh

Chaos-engineering experiments for LedgerLens, run with
[Chaos Mesh](https://chaos-mesh.org/). Each YAML in this directory injects one
fault into the `ledgerlens` namespace; `verify_experiment.py` checks that the
system recovers afterwards.

## Experiment definitions

| File | Kind | What it simulates |
| --- | --- | --- |
| `pod-kill-api.yaml` | `PodChaos` (`pod-kill`, `mode: one`) | Kills one API pod every 10 minutes for 30s — validates that the API Deployment reschedules and traffic recovers. |
| `pod-kill-ingestion.yaml` | `PodChaos` (`pod-kill`, `mode: one`) | Kills one `ingestion-worker` pod every 10 minutes for 30s — validates that ingestion resumes after a worker is lost. |
| `network-partition-ingestion.yaml` | `NetworkChaos` (`partition`, `direction: to`) | Partitions the API pods from the `ingestion-worker` pods for 60s — validates graceful degradation when the API cannot reach ingestion. |
| `network-partition-redis.yaml` | `NetworkChaos` (`partition`, `direction: to`) | Partitions the API pods from Redis for 60s — validates the feature-store / cache fallback path when Redis is unreachable. |

All experiments are created in the `chaos-mesh` namespace and select workloads
in the `ledgerlens` namespace by `app.kubernetes.io/*` labels.

## Running an experiment end to end

1. **Deploy** — apply one experiment:

   ```bash
   kubectl apply -f chaos-mesh/pod-kill-api.yaml
   ```

2. **Observe** — while the fault is active, watch pods, dashboards and logs:

   ```bash
   kubectl get pods -n ledgerlens -w
   kubectl describe networkchaos,podchaos -n chaos-mesh
   ```

   The `pod-kill-*` experiments run on a cron (`@every 10m`); the
   `network-partition-*` experiments run once for their `duration` (60s).

3. **Verify** — run immediately after applying, so sustained traffic flows
   through the fault; the script asserts the experiment's SLOs:

   ```bash
   # Against a real target (Kubernetes-hosted staging, port-forward, etc.)
   python chaos-mesh/verify_experiment.py --experiment pod-kill-api.yaml \
     --url https://ledgerlens.staging.example

   # Local default: http://localhost:8000/health
   python chaos-mesh/verify_experiment.py
   ```

4. **Clean up** — delete the experiment:

   ```bash
   kubectl delete -f chaos-mesh/pod-kill-api.yaml
   ```

## SLO pass/fail criteria

`verify_experiment.py` drives sustained traffic (4 concurrent clients) for the
traffic window while polling `GET /health`, then fails (exit `1`) if any
criterion below is violated. The criteria live in `EXPERIMENT_SLOS` in the
script — keep this table in sync.

| Experiment | Recovery ≤ | Error-rate ceiling | Silently dropped | Drain budget | Traffic window |
| --- | --- | --- | --- | --- | --- |
| `pod-kill-api.yaml` | 60s | 5% | 0 | 30s | 60s |
| `pod-kill-ingestion.yaml` | 90s | 1% | 0 | 30s | 60s |
| `network-partition-ingestion.yaml` | 60s | 5% | 0 | 10s | 90s |
| `network-partition-redis.yaml` | 60s | 2% | 0 | 10s | 90s |

Each traffic request is classified as:

- **ok** — HTTP < 400;
- **retriable** — HTTP 429/502/503/504, or the connection was refused / timed
  out before the request was sent (a client can safely retry);
- **dropped** — anything else: connection reset or read timeout after the
  request was sent, or a non-retriable error status. This is a *silently
  dropped* request.

- **Recovery** — seconds until `/health` returns 200 `{"status": "ok"}`.
- **Error rate** — `(retriable + dropped) / total`.
- **Drain time** — latency of the slowest request that still completed during
  the window, i.e. how long in-flight work took to drain. Exceeding the budget
  logs an `ALERT chaos drain budget exceeded` line and, if `--alert-webhook`
  (`CHAOS_ALERT_WEBHOOK`) is set, POSTs `{"alert": "ChaosDrainBudgetExceeded", ...}`.

| Option | Env var | Default | Purpose |
| --- | --- | --- | --- |
| `--experiment` | `CHAOS_EXPERIMENT` | `pod-kill-api.yaml` | Experiment whose SLOs to assert. |
| `--url` | — | — | API base URL; health URL becomes `<url>/health`. |
| `--health-url` | `HEALTH_URL` | `http://localhost:8000/health` | Health endpoint to poll. |
| `--traffic-url` | `TRAFFIC_URL` | health URL | Endpoint receiving sustained traffic. |
| `--traffic-seconds` | — | per experiment | Traffic window; `0` disables traffic assertions. |
| `--timeout` | `HEALTH_TIMEOUT_S` | per experiment | Override the recovery SLO. |
| `--alert-webhook` | `CHAOS_ALERT_WEBHOOK` | — | Alert target for drain-budget breaches. |
| `--metrics-url` | `METRICS_URL` | `http://localhost:8000/metrics` | Logged for reference. |
| `-v` / `--verbose` | — | off | Log every failed poll attempt at DEBUG. |

The `chaos-staging.yml` workflow runs the script after injecting the
experiment, so an SLO violation fails the CI job.

## Graceful-shutdown contract (API)

On `SIGTERM` (e.g. a pod kill) the API (`api/main.py` lifespan):

1. Stops accepting new work — every non-`/health*` request gets
   `503 Server is shutting down.` with `Retry-After: 5` (retriable).
2. Drains in-flight requests for up to `SHUTDOWN_TIMEOUT` seconds (default 30),
   then closes WebSockets, checkpoints SQLite and closes Redis.
3. Logs `[shutdown] drain_seconds=<n>`; if draining exceeds
   `SHUTDOWN_DRAIN_BUDGET_S` (default 25) it logs
   `[shutdown] ALERT drain time ... exceeded budget ...` at WARNING for
   log-based alerting.

Clients therefore see every request either complete or receive a retriable
error — never a silently dropped response. `pod-kill-api.yaml` verifies this
with zero tolerance for dropped requests. The pod's
`terminationGracePeriodSeconds` must exceed `SHUTDOWN_TIMEOUT`.

## Toxiproxy profiles (local chaos suite)

`toxiproxy.json` defines the proxies used by `tests/chaos/` (`make test-chaos`):

| Proxy | Listen | Upstream | Profiles / documented degraded behaviour |
| --- | --- | --- | --- |
| `horizon_proxy` | `:18000` | Horizon | 500ms latency — scoring p99 stays < 2s. |
| `horizon_partition` | `:18001` | Horizon | Partition — circuit breaker opens. |
| `redis_proxy` | `:16379` | Redis | Outage — cache fallback path. |
| `soroban_rpc_proxy` | `:18002` | Soroban RPC (testnet) | 3s±1s latency spike, 2s timeout, connection reset (proxy disabled) — `integrations/contract_client.py` submissions fail with `SorobanSubmissionError` within a bounded time, and after the breaker threshold further submissions fail fast with `SorobanCircuitOpenError` (`test_soroban_rpc_degradation.py`). |
| `oracle_node_proxy` | `:18003` | oracle-node stand-in (Redis PING) | 500ms latency — quorum still reached; full timeout — `detection/oracle_coordinator.py` reports an invalid quorum within a bounded time and does not submit; partial outage — quorum reached from healthy nodes (`test_oracle_degradation.py`). |

## Related

- `helm/chaos-mesh-values.yaml` — Helm values used to install the Chaos Mesh
  controller itself (service account `chaos-mesh`, UI disabled).
- `.github/ISSUES/ISSUE-117.md` — the broader "Chaos Engineering Test Suite"
  tracking issue.
- `ROADMAP.md` — chaos testing under the resilience workstream.
