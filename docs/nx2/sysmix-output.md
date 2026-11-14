# Sysmix output queue continuity

The November 11 Star Fox desktop investigation reproduced brief audio gaps with
48 kHz stereo output and SDL/PipeWire callbacks requesting 1024 frames. A steady
cutscene callback requested 1024 frames but received only 1008 queue-backed frames,
inserting 16 frames of silence (0.333 ms). Host timer lateness stayed below 1 ms
and guest PCM remained available, so this did not require a missed 5 ms timer tick.

## Cause and repair

`SysmixOutput::Pump` previously admitted at most one 240-frame block per 5 ms tick.
The sink already allowed enough queued blocks for the largest host callback plus
one guest block, but the producer did not maintain that headroom. At an unlucky
phase, delaying producer deadline 4080 by 32 frames (0.667 ms) leaves only 1008
frames for the host request at 4096. The real producer/consumer regression test
reproduces the inserted zero with the original implementation.

The output pump now refills available whole PCM blocks up to the existing adaptive
queue cap. It snapshots the available slots before the loop, so concurrent host
consumption cannot extend a timing callback indefinitely. The initial cap remains
three blocks; a 1024-frame host callback raises it to six. Full queues return
before rereading or converting guest samples. Each admitted block retains the
existing cursor checks and notification handling. The null output sink has no host
callback or queued audio to consume, so it limits each pump to one block. This
preserves 240 frames per 5 ms tick and prevents early guest-space notifications
when audio output is disabled.

A partial guest tail is deferred while queued audio remains. The existing
empty-host path can pad a real underrun, but the refill never invents extra silent
blocks merely to fill headroom. Input capture keeps its one-block-per-tick cadence.
The queue remains bounded by both its existing block limit and sample-ring capacity.

## Validation and diagnostics

The regression checks ordered stereo PCM across guest-ring wrap with no producer
jitter, 0.667 ms lateness, and 5.667 ms lateness with the missed deadline skipped.
A separate case drains the host during admission and checks that the original
refill budget still bounds the loop. A real null-sink regression checks one-block
consumption, read-cursor publication and notification thresholds across six ticks.
Existing partial-block, ownership, fault, notification and microphone tests also
pass.

Run the focused tests with:

```sh
build/bin/tests '[sysmix],[sysmix_sink]'
```

`EDEN_NX2_AUDIO_LEVEL=1` retains the existing peak report and adds two-second sink
counters: callbacks, requested frames, queue-backed rendered frames, missing
frames, underrun callbacks and maximum missing frames in one callback. The counters
are disabled by default. Explicit pause, shutdown and inactive streams do not
contribute. Startup can legitimately lack audio before the guest begins supplying
it; compare uninterrupted steady-state windows separately from loading, captures
and debugger pauses. A zero run in a monitor recording alone does not prove a
queue underrun; correlate it with these counters.

The sysmix `full` counter now includes routine ticks that find the deliberately
filled queue. `retried` remains zero for these ticks because no PCM was copied.

Local desktop validation on an integration based on `e860139f53`, also containing
separate rendering fixes, recorded 120 seconds of the opening
cutscene and 60 seconds of flight training at 48 kHz. Both recordings had no
stereo zero runs of at least four samples, no held runs of at least twelve samples,
and no clipped samples after the recording startup exclusion. Full two-second
sink windows within those recordings delivered 8,470,528 requested frames with
zero missing frames or underrun callbacks. The cutscene PipeWire trace showed no
new device or application errors. A separate 30-second training timing sample
measured 59.80 FPS, 16.72 ms mean, 17.63 ms p99 and no reported stutters. These
results cover the opening and training, not a complete mission playthrough.
