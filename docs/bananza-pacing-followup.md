# Bananza pacing follow-up

The profile proves a one-present delay in applying a changed pacing period. It does
not establish why the game's simulation sometimes runs very fast or slowly. The fix
below corrects that period transition without changing guest time or the existing
late-frame catch-up policy.

## Evidence from the owner profile

The input is `profile-run.txt`, build `HEAD-ba2d291bae-HEAD`, with SHA-256
`f499a506ae52612c88d1acff688af690da3e7db81907b39d34989b9db20f99fe`.
Line numbers below refer to that file. The final line is truncated.

- All 84 complete `DKB frame` samples between 25.126036 and 110.568445 seconds
  report speed limiting enabled, speed 100%, no forced interval, and a 60 Hz request.
  The sampled requested intervals are 1 on 58 samples, 2 on 20, and 3 on 6.
  The unthrottled swap-word logs record 40 changes after the first request.
- All 82 renderer samples report `host_pacing_mode=0`, the host Auto setting.
  The log contains no `DKB host_pace` records, consistent with that setting.
- There are 41 samples with a positive requested sleep. Their measured sleep exceeds
  the request by at most 0.023 ms. This does not support a large sleep overshoot in
  those samples, but the sampling cannot exclude stalls or setting changes between them.
- Line 975 reports present 159 changing from interval 3 to interval 2. Line 976
  reports a new `target_ms=33.333`, yet `sleep_requested_ms=47.321` and
  `sleep_actual_wall_ms=47.330`. The old period was 50 ms. This is the stale deadline
  described below, not a long OS sleep relative to the actual request.
- Line 2186 reports present 3111 at interval 1, a 56.618 ms entry-to-entry interval,
  and no pacing sleep. That particular slow present is not delayed by `PaceToVblank`.
  The entry-to-entry interval is still not a measurement of simulation speed.

The native-event records establish that slot 0 is bound to vsync at lines 442–444.
Slots 1, 2, and 3 are bound to change notifications at lines 595–601. No callback
lateness or guest event-consumption trace exists in this profile. An absent callback
trace cannot establish that vsync delivery was regular.

## Proven period-transition error and narrow fix

Previously, `PaceToVblank` waited on `m_next_vblank`, then advanced that timestamp by
the current request's period. The timestamp used for the current wait had therefore
already been calculated using the previous request's period.

For a previous phase deadline of 1 second and a present arriving 2.679 ms later:

| Request change | Old wait deadline | Corrected current-period deadline |
| --- | --- | --- |
| 3 to 2 vblanks at 60 Hz | 1.050000000 s | 1.033333333 s |
| 1 to 3 vblanks at 60 Hz | 1.016666666 s | 1.050000000 s |

The first row reproduces the logged 47.321 ms requested wait using the log's rounded
values. Applying the current period gives a 30.654333 ms wait in that deterministic
example. This is a reconstruction of the deadline math, not a replay of unlogged
intermediate presents.

`Nx2PresentPacer` now stores the previous phase deadline rather than a next deadline
calculated with an obsolete period. Its next deadline is the previous phase plus the
current computed period. `PaceToVblank` calls this helper directly. The swap-word
decoding, refresh fallback, forced-interval setting, speed percentage, guest-kernel
sleep, and its existing 2 ns adjustment are unchanged. A change in refresh or speed
percentage also takes effect on the current computed period.

For a constant period, the algorithm remains equivalent to the old code. It still
reanchors to now when more than one whole period late relative to its pending deadline.
It also still permits bounded catch-up when late by less than or exactly one period.
For example, with a 10 ms period and phase 100 ms, two calls at 120 ms can both return
without sleeping. The focused tests preserve this behavior deliberately. Removing
catch-up would be a separate pacing-policy change, not part of the proven transition fix.

The change can eliminate an overlong or short wait at a period transition. It does
not establish or fix a sustained simulation-rate error. Rendering cadence, present
returns, and the game's simulation step are different measurements.

## Vblank catch-up and guest time

`StartVsyncTimer` uses a separate CoreTiming looping event, normally at 60 Hz.
`EDEN_NX2_VSYNC_HZ` can override that experimental event rate. Guest present intervals
do not change its rate. The display-refresh query still reports 60 Hz.

`CoreTiming::Advance` reschedules a looping event from its scheduled time, not its
actual delivery time. It keeps processing due events. If a 10 ms event is first
processed at 35 ms, the 10, 20, and 30 ms callbacks are all due before the next 40 ms
deadline. The vi2 callback returns no interval override, so it retains this behavior.
The existing pause path separately rebases events scheduled into a pause.

`KReadableEvent::Signal` coalesces signals while an event is already signaled. Thus
several late callbacks are not a queued count of several guest vblanks. If the guest
clears the event between callbacks, it can nevertheless observe closely spaced
signals. The code permits this sequence; the owner profile does not show whether it
occurred. Neither the general timer nor vi2 event rescheduling is changed here.

The profile reports multicore mode, `sync_core_speed=false`, and a guest counter
frequency of 31,250,000 Hz at boot, lines 99, 106, and 295. Both the SVC system-tick
read and the Dynarmic counter callback use `CoreTiming::GetClockTicks`. The global
event clock and kernel sleep deadlines use global nanoseconds. They are independent
of the number of presents. `GetClockTicks` also has existing CPU-clock and optional
speed-sync adjustments; a boot-frequency line alone does not prove every later tick
delta. This change does not alter any of those clock sources, adjustments, or policies.
The separate `SpeedLimiter::DoSpeedLimiting` returns immediately in multicore mode.

## Additional read-only evidence in the next trace

`EDEN_DKB_FRAME_TRACE=1` now also emits `DKB vsync` at most once per second per vi2
session. The callback continues to signal the same events and returns the same result.
All new fields are diagnostic only:

- `present` is the latest entered present counter for this session, not an event/frame pairing.
- `callbacks` counts callback invocations since the previous record, not guest wakeups.
- `bound_events` is the current number of bound event objects.
- `interval_ns` is the configured looping-event period.
- `late_callbacks` counts callbacks at least one configured period late.
- `max_late_ns` is the largest callback lateness in the reporting window.
- `min_gap_ns` is the smallest observed callback-to-callback gap in the window,
  including the gap into its first callback. It is `-1` before a preceding callback exists.
- `observed_global_ns` is scheduled callback time plus its supplied lateness.
- `guest_ticks` samples the existing system-tick source when the record is emitted.

The first record can cover only one callback. Later callback counts and gaps can
identify catch-up delivery windows. They still do not prove that the guest consumed
every signal. Successive global-time and guest-tick samples can check their rate
relationship without changing it. A pause can produce a large lateness sample, so it
must not be mistaken for an ordinary scheduling stall without matching session context.

## Upstream provenance

Upstream `nx2` was fetched and inspected at
`d3a659982fff030e3e57ce15495765064d72e730` before implementing the fix.

- `f8d38607b6cf6e35ee3daad7e34e590e19421c79` introduced per-request swap-interval
  pacing to avoid forcing interval-2 titles to run at interval 1. Its next-deadline
  representation has the same transition lag and remains in the inspected upstream.
- `e7973dae7e2ea44665ff4ec83d7214a5d2daa366` replaced host sleeps with guest-kernel
  sleeps so the service does not hold guest core 3 through the pacing wait. Retained.
- `d5c8b1c` separates vi2 event sources. Bound change events are not periodic vsync
  events. Retained, including the existing slot-binding behavior.
- `0218eacfd560d0ce48b3ffe64d279ee85bbf1347` established the native Switch 2
  31.25 MHz guest counter. Already present and unchanged.
- [Upstream PR 403](https://github.com/Djihads80/Eglantine/pull/403), head
  `8f70204825c2ea7fb91d86fac196ad93c22e885d`, was merged as
  `5ae14067e339395275461a725bbcbb8f7c41b56c` on November 4, 2026. It patches
  Mario Kart World's own mode selection and frame-time measurement, then selects
  interval-1 pacing only when that title's dynamic patch succeeded. It is not a
  general-purpose pacing fix, does not repair the deadline-transition math, and
  provides no supported Bananza patch. It is not applied here.

## Verification and remaining limit

`[nx2_present_pacer]` exercises the production helper, including shorter and longer
period transitions, repeated changes, late frames, reanchoring, and fixed-period
equivalence with the previous algorithm. With the old production math extracted
unchanged, three initial transition tests failed. With the fix and an added
late-transition case, all six tests and 82 assertions pass. No game was run for
this investigation.

Before the next upstream merge, the atomic-and-pacing tree built both `lichii-cmd`
and `tests` with Clang 20 and Ninja using all eight available cores. All 496 test
cases passed with frame tracing disabled and with `EDEN_DKB_FRAME_TRACE=1`.
The combined atomic/pacing filter passed 10 cases and 13,800 assertions.

The remaining simulation-speed claim needs a trace tied to a clearly identified
fast or slow gameplay interval, including the new vsync fields. Guest simulation-step
measurements would be needed to attribute its rate to the game's own mode logic.
The current evidence does not justify a title-specific FPS override, a dynamic-FPS
patch for Bananza, or scaling native guest time to measured rendering throughput.
