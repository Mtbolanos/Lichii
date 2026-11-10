<!--
Fritchii's "What's new" notes for this build: paste the release notes (the GitHub release body,
e.g. E:\OotRemakeEmulation\release\fritchii-vX\notes.md) here before the release build. The
download instructions at the top are dropped in the app. While this file holds only this comment,
no "What's new" window pops up after an update (Help > What's New says there are no notes yet).
-->
## ⬇️ Download this file: **`Fritchii-v5.6-Windows-x64.zip`**

On Fritchii v4 to v5.5 you don't need to download anything: it offers this update when you open it. Otherwise unzip it anywhere and run `Fritchii-v5.6\lichii.exe`. Don't download "Source code (zip)" or "Source code (tar.gz)": they contain no emulator.

---

Fritchii v5.6.

**Heads up: "Preparing shaders" runs once more** (around 15 minutes on an 8-core CPU), on every GPU, because of the AMD horizon fix below and a new detail Fritchii now stores about each shader. After that it's straight into the game again. The first minute or two after it can also be a bit slower while the driver settles.

#### Fixed
- **The huge glowing triangles ("light beams") are much rarer.** In Kokiri Forest, Kakariko, the graveyard and Hyrule Field they used to grow over a session. One cause is fixed: a compute shader wrote further than the emulator tracked, and the game read old data back from there. Every compute shader is now checked once, and tracked fully if it writes further. No fps cost. They can still show up in long sessions: if you see one, use Help > Export Log and post it in the Issues channel.
- **AMD: no more GPU hangs on draws the game makes with mesh shaders** (zortos293/Lichii#814). AMD's driver now goes through a compute fallback for them. New **Mesh Shaders** setting in Graphics (Automatic, Native, Compute) if you want to try the other path.
- **AMD: no more rainbow "disco" outline** where mountains and far scenery meet the sky. Draws that don't output every colour channel left the rest undefined, and AMD's driver filled it with garbage. Port of jpkto's Lichii #353 (Fritchii #29 by trushna88).
- **The opening cutscenes and the main menu video stay in sync with their audio.** The game only shows about 30 frames a second on those screens, so the 60 fps videos played at half speed while the music kept going (almost 7 seconds behind after 12 seconds). Videos now show the frame that matches the audio and skip the rest. A broken video frame or a failing decoder can no longer crash the game.
- **The Song of Time no longer ends with a loud crackle** (zortos293/Lichii#1073 by justin-wq1337).
- **No more freezes of up to 3 seconds** when the game swaps a compute shader mid-play (zortos293/Lichii#969 by El-Sashok).
- **Unlocking the speed limit no longer makes OoT run fast.** Above 60 fps the whole game sped up; it now stays at its normal speed. Turbo and Slow still work as fast-forward and slow motion. In OoT, Ctrl+U now switches the high refresh rate mode instead (see New).
- **Opening the ocarina without a microphone no longer crashes the game** (zortos293/Lichii#497 and #881, via Fritchii #16).
- Mods that patch the same spot as Fritchii's built-in patches win over them (Fritchii #5). Game titles from NUP files show properly (Fritchii #20). Empty controller mapping values are no longer lost (Fritchii #30 by FFcustoms). Fritchii builds on Linux again (Fritchii #19).
- **Help > Export Log always sends this session's log.** A Fritchii that never finished closing (for example after an update) could keep the log file locked, and the export then sent an old log.

#### New
- **High refresh rate (experimental): press Ctrl+U in OoT to go above 60 fps** at normal game speed, and again to go back, no restart needed. The cap is in Graphics > Advanced (your monitor's refresh rate, 90, 120, 144 or 165), and the Graphics checkbox follows Ctrl+U. Water ripples keep moving above 60 too (zortos293/Lichii#1152, after #655 by atk3001). It's new: tell us if a timer, cutscene or minigame acts weird with it on. The 30 FPS mode still wins over it.
- **Dark's standalone Dynamic FPS mod is skipped and hidden automatically**, since Fritchii has it built in (the mod also blocked the high refresh mode). Your files aren't touched. If you turned the built-in off in Add-Ons to use the mod, nothing changes for you.
- **Reset to Defaults** (Settings > General) puts every setting back to Fritchii's own tuned settings, after saving a backup. **Restore Settings** brings any saved set back. Controls and game folders stay.
- **"My settings don't apply" explained:** when OoT has its own values in Configure Game, the Settings window now says so at the top, with one button to put OoT back on your global settings. Export Log lists them too.
- **RAM and VRAM in the performance overlay**, with how much your PC has left (amber when it's getting low).
- **"Shaders build again in game" reports explain themselves:** the log now shows how big the GPU driver's own shader cache is at boot, after preparing and at exit, whether the last session closed cleanly, and how fast preparing ran.
- **OoT's videos look like they should again on NVIDIA RTX.** The title screen and cutscene videos go through the game's own upscaler again (it removes banding and sharpens), instead of the plain stretch every build has used since v5. It costs nothing visible on RTX cards. GPUs that would have to emulate its math keep the fast stretch. Graphics > "Ocarina of Time movies" lets you pick either.
- **Discord and Donate buttons** on the home screen and in the Help menu.

#### Faster and lighter
- **About 1 GB less RAM:** memory the game cleared to zero is given back to Windows.
- **PCs with less than 24 GB of RAM** keep fewer old GPU pipelines around, so long sessions page less.
- **About 5 fps more in Castle Town** (45 to 50 on an RTX 4080 standing in the market): the emulator stopped sending the GPU thousands of empty submissions a second, and reports and fences no longer allocate memory each time (part of zortos293/Lichii#1161).
- **No more VRAM pile-up:** old render targets the game replaced are cleaned out, so VRAM stays around 5.5 GB in Castle Town instead of filling to 10-11 GB, and the freezes when it got full are gone (fights especially).
- **Long sessions:** pipelines rebuilt after being cleaned out keep their buffers, so a long session can't slowly break some meshes (zortos293/Lichii#1143).
- **Fewer dips in busy spots:** a depth buffer that the game clears right away is no longer copied in from game memory first (7.5 MB every frame in Castle Town, the cause of an NVIDIA player's dips). About +2.5 fps in Castle Town.
- **Texture memory on 10 GB and bigger cards** can fill further before the cleanup starts (zortos293/Lichii#1129 by Liomar Oliveira).
- **FXAA is off by default.** OoT's own anti-aliasing (or DLSS) already smooths edges, FXAA on top only blurred the picture. Fritchii's tuned settings are applied once more with this update; you can change anything afterwards.

### In v5.5

- **No more flickering camera for players who had frame generation on before v5.** Fritchii has no frame generation, but an old setting kept it running in the background (and turned off Fritchii's frame pacing). It can't run anymore.
- **Extended Dynamic State is set back to EDS2 once.** Players who set it to Disabled or EDS3 (often an old AMD crash tip) had "Preparing shaders" skip almost every shader. EDS2 is what the shaders are recorded with, and the AMD crash that tip avoided is fixed since v5.3. You can still change it afterwards.
- **NVIDIA: the shader cache size is set by Fritchii itself.** Newer drivers no longer show "Shader Cache Size" in the NVIDIA Control Panel. Fritchii now tells the driver to keep a large cache for Fritchii only, nothing else on your PC changes.
- **DLSS fixes:** the "Use NVIDIA DLSS" checkbox could be overridden by an old per-game value and show unchecked in game; it's a global setting now. New **DLSS preset** option (Default, J, K recommended, L, M) in Graphics.
- Graphics > Extensions now says it: GPU thread pinning is built in and on automatically.
- The old "Prepare Shaders" window says it's the optional quick warm-up (the full preparation runs by itself when you start the game).

### In v5.4

- **AMD: "Preparing shaders" now prepares the whole game.** Until now it skipped 116,373 of OoT's 120,461 shaders on AMD (it finished in seconds) because they were recorded on an NVIDIA card, which handles one GPU setting (the dynamic logic operation) differently. Fritchii now converts them for AMD's driver: all of them get prepared, so new areas play like you've been there before, as on NVIDIA. Thanks to koriko for the log that pinned it.
- **AMD: expect a full "Preparing shaders" screen once** (around 15 minutes on an 8-core CPU), even if you went through it on v5.x. After that it's straight into the game.
- Nothing changes on NVIDIA.

### In v5.3

- **AMD: Shader libraries are really off by default now.** v5.2 meant to turn them off on AMD's Windows driver but its check never matched, so RX 5000/6000 cards kept crashing while loading shaders. If you unticked the setting yourself, nothing changes for you.
- **Updates retry by themselves.** When GitHub's file server answers with an error like 504 or the connection drops, Fritchii now tries again a couple of times before giving up (from the next update on).

### In v5.2

- **GTX 10 series and older: invisible geometry is drawn again.** Objects the game draws with mesh shaders, which these cards lack, now go through a compute fallback (zortos's prototype, zortos293/Lichii#569, #642, #841). Some foliage or walls may still be missing or flicker.
- **AMD: textures no longer write into other textures' memory.** Images whose size isn't a multiple of their block size wrote past their edge while being decoded. NVIDIA ignores that, AMD put it in other textures: black blocks, a crosshatch, and in the end a GPU crash. From CheezerWheezer's AMD fork.
- **AMD (Windows driver): Shader libraries are off by default.** RX 5000 and 6000 cards crashed building them; unticking the setting was the workaround, now it's automatic. NVIDIA keeps them.
- **The full right-click menu is back on the home screen** (Open Mod Data Location, Dump RomFS, Remove, Verify Integrity, Start Game without Custom Configuration...), for modders.
- **No more crash when the PC's memory budget is reached** during long sessions (seen in Castle Town after half an hour). Buffers in system memory may now go over Windows' budget, like textures already could. Also from CheezerWheezer's AMD fork.

### In v5.1

- **Dialogue text no longer turns into scraps of other textures.** Each new text box size could show garbage the first time it appeared.
- **Delete Shader Cache** in the Help menu (and when you right-click the game on the home screen), with a confirmation. Only for a cache that went bad: the shaders are prepared again on the next start.
- **AMD: likely fix for a crash right after pressing Play** (seen on an RX 6600 with a shader cache from v4). v5's change for the red line in the Royal Family's Tomb is taken back for now, so that line can show again on AMD until it's redone and tested on AMD hardware.
- One v5 change to how GPU memory is refreshed after small writes is taken back too: in this emulator it could bring back stale data.
- **4K and scaled displays:** Fritchii follows the scale set in Windows, as far as your screen leaves room for its windows. 150% on a 4K screen used to become 200% (windows too big, dialogs cut off) and 125% became 100% (tiny text).
- **Game Settings fits the screen:** long tabs scroll, the window resizes, and OK / Apply are always reachable.

### Everything new in v5

Windows x64. Tested with The Legend of Zelda: Ocarina of Time (0400486024DEC000) on a Ryzen 7 7800X3D + RTX 4080: stable 60 fps in a lot of places, rarely under 45 even in the heaviest ones (Water Temple included), no shader stutter after the preparation.

#### New
- **Shaders are prepared once, before you play.** The first time you start OoT, a "Preparing shaders" screen builds the game's shaders at full speed (about 15 minutes on an 8-core CPU, once). After that, places you visit for the first time play like you've been there before: walking into a new area used to freeze for ~0.7 s per new shader, now it's 40-70 ms, and loads are shorter. A new driver or GPU prepares them again. Built on zortos's precompile work (zortos293/Lichii#895, #896).
- **New look.** Graphite theme, a home screen, an in-app "What's new", and Help > Export Log for bug reports.
- **In-game controls.** Move the mouse to the top of the game for Pause, Restart, Stop, Fullscreen and the menus.
- **New performance graph** (View > Performance Overlay): big fps, frame time, the lowest recent fps and a line against 60 (or 30). Loading screens leave a gap instead of a dip.

#### Faster
- **Moving NPCs and crowds no longer drag the frame rate down.** About +3 fps in Castle Town from each of the fixes behind it (render target relocation, image pool, an exact frame clock, cleanup of stale copies), and ~2.4 GB less video memory (8.8 to 6.4 GB in Castle Town).
- **~2.3 GB less RAM**, so 16 GB PCs have more room.
- **Windows 11: a precise game clock** built in. Players who got big gains from the GlobalTimerResolutionRequests registry tweak get it without the tweak.
- Shader translation moved off the GPU thread.
- Retrained profile-guided build on the new code.

#### Fixed
- **New players: "Press Start", chat bubbles and lighting** are right on an empty shader cache (they were missing or wrong while shaders were still building).
- **Objects that could vanish for a whole session** in some games (clip distances, zortos293/Lichii#902).
- **Fastmem kept on PCs with little free memory** instead of silently running 20-25% slower (zortos293/Lichii#645).
- Less RAM while shaders are prepared (#878), a rare channel deadlock (#609), mods folders that don't exist (#686), Windows long paths (#576), and a few smaller ones (#918, #727, #610).

#### Shader cache
- The cache moves to version 54 (zortos's numbering), so caches from zortos's builds and Fritchii's are shared. The first start copies your old cache once (about +1.2 GB for OoT). The old `v52` file stays so you can roll back to v4; you can delete it later if you don't plan to.

#### Known
- AMD and Intel changes could not be tested on that hardware here: please report problems with your GPU model and the log (Help > Export Log).

Dynamic FPS by Dark (built-in). Thanks to zortos293 and every Lichii contributor whose work is in here, and to the **Ocarina of Forks** Discord for the moral support, the novel ideas and the debugging help.
