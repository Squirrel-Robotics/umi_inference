#!/usr/bin/env python3
"""OpenPI WebSocket client tuned for a small, jittery robot LAN."""

from __future__ import annotations

import socket
import time
from typing import Any

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import websockets.sync.client

from camera_transport import TRANSPORT_PROBE_SCHEMA
from camera_sync import CLOCK_SYNC_SCHEMA


PING_INTERVAL_SECONDS = 0.05
PING_TIMEOUT_SECONDS = 5.0
TRANSPORT_WARMUP_COUNT = 3
TRANSPORT_WARMUP_BYTES = 12 * 1024
CLOCK_SYNC_PROBE_COUNT = 5
CLOCK_SYNC_PROBE_BYTES = 1
# The deployment camera contract is 50 ms end to end.  Half is reserved for
# clock uncertainty; the policy subtracts the measured uncertainty from the
# remaining timestamp-skew budget, so the total bound still stays at 50 ms.
CLOCK_SYNC_MAX_UNCERTAINTY_MS = 25.0


class ClockSyncUncertain(RuntimeError):
    """A transient high-RTT clock sample that should be retried, not fatal."""

    reason = "clock_sync_uncertain"

    def __init__(
        self,
        *,
        uncertainty_ms: float,
        max_uncertainty_ms: float,
        best_rtt_ms: float,
    ) -> None:
        self.uncertainty_ms = float(uncertainty_ms)
        self.max_uncertainty_ms = float(max_uncertainty_ms)
        self.best_rtt_ms = float(best_rtt_ms)
        super().__init__(
            "XR/5090 clock offset is not precise enough for camera alignment: "
            f"uncertainty={self.uncertainty_ms:.3f}ms "
            f"limit={self.max_uncertainty_ms:.3f}ms "
            f"best_rtt={self.best_rtt_ms:.3f}ms"
        )


class LowLatencyWebsocketClientPolicy(WebsocketClientPolicy):
    """Disable proxies and keep TCP active between asynchronous chunks."""

    def _wait_for_server(self):  # noqa: ANN202
        headers = (
            {"Authorization": f"Api-Key {self._api_key}"}
            if self._api_key
            else None
        )
        connection = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            proxy=None,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
        )
        raw_socket = getattr(connection, "socket", None)
        if raw_socket is not None:
            try:
                # DSCP EF maps to the voice access category on WMM-capable APs.
                raw_socket.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 46 << 2)
                raw_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        metadata = msgpack_numpy.unpackb(connection.recv())
        return connection, metadata


def warm_transport(
    policy: WebsocketClientPolicy,
    *,
    count: int = TRANSPORT_WARMUP_COUNT,
    payload_bytes: int = TRANSPORT_WARMUP_BYTES,
) -> list[dict[str, Any]]:
    """Grow the client TCP window before any timestamped observation exists."""

    if count <= 0 or payload_bytes <= 0:
        raise ValueError("transport warm-up count and payload size must be positive")
    results: list[dict[str, Any]] = []
    padding = bytes(payload_bytes)
    for sequence in range(1, count + 1):
        started = time.monotonic()
        response = policy.infer(
            {
                "_transport_probe": {
                    "schema": TRANSPORT_PROBE_SCHEMA,
                    "sequence": sequence,
                    "padding": padding,
                }
            }
        )
        probe = response.get("transport_probe")
        if (
            not isinstance(probe, dict)
            or probe.get("schema") != TRANSPORT_PROBE_SCHEMA
            or probe.get("sequence") != sequence
            or probe.get("payload_bytes") != payload_bytes
        ):
            raise RuntimeError(f"invalid transport warm-up response: {response}")
        results.append(
            {
                "sequence": sequence,
                "payload_bytes": payload_bytes,
                "round_trip_ms": (time.monotonic() - started) * 1000.0,
            }
        )
    return results


def estimate_server_clock_offset(
    policy: WebsocketClientPolicy,
    *,
    count: int = CLOCK_SYNC_PROBE_COUNT,
    payload_bytes: int = CLOCK_SYNC_PROBE_BYTES,
    max_uncertainty_ms: float = CLOCK_SYNC_MAX_UNCERTAINTY_MS,
) -> dict[str, Any]:
    """Estimate ``server_wall_time - client_wall_time`` without policy work.

    Every sample uses the existing transport-only request path, which never
    reads cameras, invokes the model, records a request, or enables robot
    output.  The minimum-delay sample is retained, as in NTP, so queuing from
    other samples cannot silently become a camera timestamp correction.
    """

    if count <= 0 or payload_bytes <= 0:
        raise ValueError("clock probe count and payload size must be positive")
    if max_uncertainty_ms <= 0.0:
        raise ValueError("clock probe uncertainty limit must be positive")
    samples: list[dict[str, Any]] = []
    padding = bytes(payload_bytes)
    for sequence in range(1, count + 1):
        client_send_wall_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        response = policy.infer(
            {
                "_transport_probe": {
                    "schema": TRANSPORT_PROBE_SCHEMA,
                    "sequence": sequence,
                    "padding": padding,
                }
            }
        )
        completed_monotonic_ns = time.monotonic_ns()
        client_receive_wall_ns = time.time_ns()
        round_trip_ns = completed_monotonic_ns - started_monotonic_ns
        wall_round_trip_ns = client_receive_wall_ns - client_send_wall_ns
        if abs(wall_round_trip_ns - round_trip_ns) > 1_000_000:
            raise RuntimeError(
                "XR wall clock changed during a camera clock probe: "
                f"wall_elapsed={wall_round_trip_ns / 1e6:.3f}ms "
                f"monotonic_elapsed={round_trip_ns / 1e6:.3f}ms"
            )
        probe = response.get("transport_probe")
        if (
            not isinstance(probe, dict)
            or probe.get("schema") != TRANSPORT_PROBE_SCHEMA
            or probe.get("sequence") != sequence
            or probe.get("payload_bytes") != payload_bytes
        ):
            raise RuntimeError(f"invalid clock probe response: {response}")

        server_receive_wall_ns = probe.get("server_receive_wall_time_ns")
        server_send_wall_ns = probe.get("server_send_wall_time_ns")
        if (
            isinstance(server_receive_wall_ns, int)
            and not isinstance(server_receive_wall_ns, bool)
            and isinstance(server_send_wall_ns, int)
            and not isinstance(server_send_wall_ns, bool)
            and server_send_wall_ns >= server_receive_wall_ns
        ):
            # Four-timestamp NTP estimate.  The WebSocket encoder/decoder time
            # remains inside the measured delay and therefore inside the
            # reported uncertainty bound.
            offset_ns = (
                (server_receive_wall_ns - client_send_wall_ns)
                + (server_send_wall_ns - client_receive_wall_ns)
            ) // 2
            server_processing_ns = server_send_wall_ns - server_receive_wall_ns
            network_delay_ns = max(0, wall_round_trip_ns - server_processing_ns)
            method = "ntp_four_timestamp_min_delay"
        else:
            # Backward compatibility with an already-running server that
            # returns one timestamp at the end of its transport-only handler.
            server_wall_ns = probe.get("server_wall_time_ns")
            if isinstance(server_wall_ns, bool) or not isinstance(server_wall_ns, int):
                raise RuntimeError(
                    f"clock probe response has no server wall timestamp: {response}"
                )
            client_midpoint_ns = (
                client_send_wall_ns + client_receive_wall_ns
            ) // 2
            offset_ns = server_wall_ns - client_midpoint_ns
            network_delay_ns = completed_monotonic_ns - started_monotonic_ns
            method = "single_server_timestamp_min_rtt"

        samples.append(
            {
                "sequence": sequence,
                "round_trip_ns": int(round_trip_ns),
                "network_delay_ns": int(network_delay_ns),
                "server_minus_client_wall_time_ns": int(offset_ns),
                "method": method,
            }
        )

    best = min(
        samples,
        key=lambda sample: (
            sample["network_delay_ns"],
            sample["round_trip_ns"],
            sample["sequence"],
        ),
    )
    uncertainty_ms = best["network_delay_ns"] / 2e6
    if uncertainty_ms > max_uncertainty_ms:
        raise ClockSyncUncertain(
            uncertainty_ms=uncertainty_ms,
            max_uncertainty_ms=max_uncertainty_ms,
            best_rtt_ms=best["round_trip_ns"] / 1e6,
        )
    return {
        "schema": CLOCK_SYNC_SCHEMA,
        "method": best["method"],
        "server_minus_client_wall_time_ns": best[
            "server_minus_client_wall_time_ns"
        ],
        "best_round_trip_ns": best["round_trip_ns"],
        "best_network_delay_ns": best["network_delay_ns"],
        "uncertainty_ns": best["network_delay_ns"] // 2,
        "sample_count": len(samples),
        "measured_client_wall_time_ns": time.time_ns(),
    }
