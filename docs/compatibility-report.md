# Opt-in desktop compatibility reports

After stopping a Switch 2 game, right-click that game and choose **Submit compatibility report**. The action is available only for the latest session in the current Lichii process. Starting another game clears that session, and closing Lichii discards it. This feature does not store play history.

Search the website catalog, select the exact game, and press **Confirm game**. Search is explicit and supports catalog pages. A Nintendo title ID is never used as a catalog ID. The search box uses control metadata or the loader's known title, including NUP package titles. Missing title metadata leaves the search blank rather than using a filename.

Review the editable experience and setup tabs, then the public JSON preview. Build, CPU, OS, RAM, title version, active renderer identity and launch settings are prefilled when available. Vulkan supplies the selected physical device model and its driver name/version using the existing vendor-specific version formatter. OpenGL supplies the active context's renderer and complete implementation/version string, not a guessed standalone driver version. Missing values remain Unknown. Prefills are not performance measurements. Compatibility and progress require the player's judgment. Settings changed during gameplay must be corrected by the player.

Both attestations start unchecked and reset after an edit or submission attempt. The player must affirm them and press **Submit report**. There is no account or embedded secret. Names, handles, report text, hardware and settings are public. The service sees the client's IP address and uses it for rate limiting. Logs and files are never read or uploaded by this feature.

**Clear draft** clears editable content and catalog confirmation. Cancel aborts an outstanding search or stops waiting for a submission. Cancellation cannot retract data already received by the server. After an uncertain response, an explicit retry of an unchanged draft uses the same UUID and body. There are no automatic retries. Closing the dialog discards that identity, so check for an accepted report before starting another draft.

## Implementation boundary

`MainWindow::BootGame` captures one allowlisted value snapshot after successful core loading and before starting the emulation thread. `OnEmulationStopped` enables the context action for that snapshot's exact local program ID. Application changes invalidate it. The dialog receives copied values, not a core object, filesystem object, settings serializer or log path.

Each graphics backend copies its device identity during construction while its selected device/context is available. The session then copies those strings before teardown; it performs no GL calls, adapter enumeration or log reading. The null backend leaves the identity empty. No session performance measurements are collected by the reporting feature.

Mods and updates use the add-on catalog's enabled entries at launch. Only their display names and version/type labels enter the draft, never their locations or file contents. Disabled entries and DLC are excluded. These are configured enabled entries, not proof that every patch applied successfully. Summaries longer than the API's 500-character limit are explicitly marked as truncated for review.

NUP packages select their embedded update internally, outside that catalog. Their effective game version comes from the loaded package's display-version metadata and is included as **Loaded NUP package** in the updates field. This does not guess whether a base-only package contains a separate update.

`CompatibilityReporting::Dialog` owns the catalog confirmation, consent state, payload validation and asynchronous network requests. Its normal constructor creates a dedicated network manager. Requests disable cookie loading/saving, credential reuse, caching and redirects, and have a 20-second deadline and 256 KiB response bound. Only an explicit Search or Submit starts a request. Server errors are shown as local messages and known field names, never arbitrary HTML. Receipt links must match the approved HTTPS origin and report UUID path.

A separate controller/view design was considered. The dialog-owned implementation avoids a second mutable representation of one fixed-purpose form. The tests inject a network manager through a private construction seam and exercise the same widgets and state transitions as production, without emulator startup.

The native v1 wire contract is [the website API reference](https://github.com/zortos293/Lichii-Website/blob/57ad91b89e80b5be040892c7e0983d69c887e307/docs/desktop-api.md). It has no private-log consent or upload fields. The two required attestations concern public disclosure and safe report contents.

## Verification without a game

Build the standalone Qt test target with clang and Ninja:

```sh
cmake -S src/tests/qt_compatibility -B build/compatibility-report -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build build/compatibility-report --parallel 4
ctest --test-dir build/compatibility-report --output-on-failure
```

The target also joins the normal Qt build when `LICHII_TESTS` is enabled. It uses the production Qt string-conversion definitions. Tests cover consent gates, catalog confirmation and pagination, stale responses, cancellation, double submission, exact allowed fields, retry identity, field errors, rate limits, malformed/oversized responses and unsafe receipt URLs. A loopback HTTP fixture checks actual Qt transport and ensures a seeded cookie is not sent. Neither fake transport nor loopback tests can send a production report.

For a synthetic dialog using only fake transport:

```sh
build/compatibility-report/compatibility_report_tests --preview
QT_QPA_PLATFORM=offscreen build/compatibility-report/compatibility_report_tests \
  --screenshot /tmp/compatibility-report.png
```

The preview preloads synthetic experience text for visual inspection. Real dialogs do not preselect a compatibility rating or populate progress and details. Do not submit synthetic fixtures to the live website to test connectivity.
