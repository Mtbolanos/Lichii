// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <thread>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define NX2_IDLE_HAS_PAUSE 1
#elif defined(__aarch64__)
#define NX2_IDLE_HAS_PAUSE 1
#endif
#include "common/nx2_hitch.h"
#include "common/nx2_timeline.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/arm/debug.h"
#include "core/core.h"
#include "core/debugger/debugger.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc.h"
#include "video_core/nx2_debug_control.h"

namespace Kernel {

namespace {

/// EDEN_NX2_IDLE_PAUSE=1 (live, VideoCore::Nx2Debug::g_idle_pause, off by default; from
/// zortos293/Lichii#641, where it is on): Idle's poll runs PAUSE instead of SwitchToThread, so
/// there is no syscall per iteration (10-16% of each guest core in Castle Town), and reads the
/// clock only every 16 pauses instead of every iteration. The poll stays bounded by time: 16
/// pauses are ~2,200 cycles on Intel Skylake and later (~140 each) and far fewer on AMD (~10-60
/// each), so the deadline overshoots by well under a microsecond. Hosts without a pause hint keep
/// the yield loop.
bool IdlePauseOn() {
#if defined(NX2_IDLE_HAS_PAUSE)
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_idle_pause,
                                            "EDEN_NX2_IDLE_PAUSE", 0) > 0;
#else
    return false;
#endif
}

void IdlePause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

/// EDEN_NX2_CORE_LOCK=1 (boot switch, off by default; from zortos293/Lichii#641, where it is on):
/// the core context lock is a KSpinLock (spins briefly, then an address wait) instead of a
/// std::mutex, and Idle sleeps on an address wait on the interrupt flag instead of a condition
/// variable; Interrupt sets the flag and signals the JIT under the lock and wakes the sleeper
/// after releasing it. This sits on the scheduler's thread-state path (OnThreadStateChanged ->
/// scheduler lock Unlock -> RescheduleCores -> Interrupt), 10-16% of each guest core in Castle
/// Town. #641 also moved the waits to its own WaitOnAddress wrapper because libstdc++'s
/// std::atomic::wait goes through winpthreads on MinGW; here std::atomic::wait is used, which on
/// MSVC's STL (the clang-cl builds) already is WaitOnAddress. Read once per core at kernel setup,
/// so a core never mixes the two locks.
bool CoreLockEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CORE_LOCK");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

/// The guest thread's nn::os name for the timeline (Common::Nx2Timeline::NameGuestThread), once
/// it has one; threads the SDK did not create keep their id.
void NameGuestThreadForTimeline(const KThread* thread) {
    const u64 thread_id = thread->GetThreadId();
    if (!Common::Nx2Timeline::WantsGuestThreadName(thread_id)) {
        return;
    }
    if (const auto name = Core::GetThreadName(thread)) {
        Common::Nx2Timeline::NameGuestThread(thread_id, *name);
    }
}

} // Anonymous namespace

/// Locks the core context with whichever lock EDEN_NX2_CORE_LOCK chose.
class PhysicalCore::ScopedGuard {
public:
    explicit ScopedGuard(PhysicalCore& core) : m_core{core} {
        if (m_core.m_core_lock) {
            m_core.m_fast_guard.Lock();
        } else {
            m_core.m_guard.lock();
        }
    }

    ~ScopedGuard() {
        if (m_core.m_core_lock) {
            m_core.m_fast_guard.Unlock();
        } else {
            m_core.m_guard.unlock();
        }
    }

    YUZU_NON_COPYABLE(ScopedGuard);
    YUZU_NON_MOVEABLE(ScopedGuard);

private:
    PhysicalCore& m_core;
};

PhysicalCore::PhysicalCore(KernelCore& kernel, std::size_t core_index)
    : m_core_index{core_index}, m_core_lock{CoreLockEnabled()}
{
    m_is_single_core = !kernel.IsMulticore();
}
PhysicalCore::~PhysicalCore() = default;

void PhysicalCore::RunThread(KernelCore& kernel, Kernel::KThread* thread) {
    auto* process = thread->GetOwnerProcess();
    auto& system = kernel.System();
    auto* interface = process->GetArmInterface(m_core_index);

    interface->Initialize();

    const auto EnterContext = [&]() {
        // Lock the core context.
        ScopedGuard lk{*this};

        // Check if we are already interrupted. If we are, we can just stop immediately.
        if (m_is_interrupted) {
            return false;
        }

        // Mark that we are running.
        m_arm_interface = interface;
        m_current_thread = thread;

        // Acquire the lock on the thread parameters.
        // This allows us to force synchronization with Interrupt.
        interface->LockThread(thread);

        return true;
    };

    const auto ExitContext = [&]() {
        // Save the JIT context back to the thread so the debugger can
        // read the current register state. Debug halt paths (step,
        // breakpoint, watchpoint) return from RunThread without going
        // through the scheduler's Unload/SaveContext.
        if (system.DebuggerEnabled()) {
            interface->GetContext(thread->GetContext());
        }

        // Unlock the thread.
        interface->UnlockThread(thread);

        // Lock the core context.
        ScopedGuard lk{*this};

        // On exit, we no longer are running.
        m_arm_interface = nullptr;
        m_current_thread = nullptr;
    };

    while (true) {
        // If the thread is scheduled for termination, exit.
        if (thread->HasDpc() && thread->IsTerminationRequested()) {
            thread->Exit(kernel);
        }

        // Notify the debugger and go to sleep if a step was performed
        // and this thread has been scheduled again.
        if (thread->GetStepState() == StepState::StepPerformed) {
            system.GetDebugger().NotifyThreadStopped(thread);
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // The timeline's guest layer names the guest threads (nn::os names, read from guest
        // memory here, outside the scheduler lock).
        if (Common::Nx2Timeline::Guest()) [[unlikely]] {
            NameGuestThreadForTimeline(thread);
        }

        // Otherwise, run the thread.
        Core::HaltReason hr{};
        {
            // If we were interrupted, exit immediately.
            if (!EnterContext()) {
                return;
            }

            if (thread->GetStepState() == StepState::StepPending) {
                hr = interface->StepThread(thread);

                if (True(hr & Core::HaltReason::StepThread)) {
                    thread->SetStepState(StepState::StepPerformed);
                }
            } else {
                hr = interface->RunThread(thread);
            }

            ExitContext();
        }

        // Determine why we stopped.
        // If a step completed successfully, skip other halt reason handlers
        // the step takes priority (e.g. step may also set InstructionBreakpoint
        // if the next instruction happens to be a breakpoint).
        const bool step_completed = True(hr & Core::HaltReason::StepThread)
                                    && thread->GetStepState() == StepState::StepPerformed;
        const bool supervisor_call = !step_completed && True(hr & Core::HaltReason::SupervisorCall);
        const bool prefetch_abort = !step_completed && True(hr & Core::HaltReason::PrefetchAbort);
        const bool breakpoint = !step_completed && True(hr & Core::HaltReason::InstructionBreakpoint);
        const bool data_abort = !step_completed && True(hr & Core::HaltReason::DataAbort);
        const bool interrupt = !step_completed && True(hr & Core::HaltReason::BreakLoop);

        // Since scheduling may occur here, we cannot use any cached
        // state after returning from calls we make.

        // Notify the debugger and go to sleep if a breakpoint was hit,
        // or if the thread is unable to continue for any reason.
        if (breakpoint || prefetch_abort) {
            if (breakpoint) {
                interface->RewindBreakpointInstruction();
                // RewindBreakpointInstruction sets the JIT state to the
                // saved breakpoint context. Update the thread context to
                // match, since ExitContext already saved the post-execution
                // state.
                interface->GetContext(thread->GetContext());
            }
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadStopped(thread);
            } else {
                interface->LogBacktrace(process);
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Notify the debugger and go to sleep on data abort.
        if (data_abort) {
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadWatchpoint(thread, *interface->HaltedWatchpoint());
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Handle system calls.
        if (supervisor_call) {
            // Perform call.
            Svc::Call(system, interface->GetSvcNumber());
            return;
        }

        // Handle external interrupt sources.
        if (interrupt || m_is_single_core) {
            return;
        }
    }
}

void PhysicalCore::LoadContext(const KThread* thread) {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->SetContext(thread->GetContext());
        interface->SetTpidrroEl0(GetInteger(thread->GetTlsAddress()));
        interface->SetWatchpointArray(&process->GetWatchpoints());
    }
}

void PhysicalCore::LoadSvcArguments(const KProcess& process, std::span<const uint64_t, 8> args) {
    process.GetArmInterface(m_core_index)->SetSvcArguments(args);
}

void PhysicalCore::SaveContext(KThread* thread) const {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->GetContext(thread->GetContext());
    }
}

void PhysicalCore::SaveSvcArguments(KProcess& process, std::span<uint64_t, 8> args) const {
    process.GetArmInterface(m_core_index)->GetSvcArguments(args);
}

void PhysicalCore::CloneFpuStatus(KThread* dst) const {
    auto* process = dst->GetOwnerProcess();

    Svc::ThreadContext ctx{};
    process->GetArmInterface(m_core_index)->GetContext(ctx);

    dst->GetContext().fpcr = ctx.fpcr;
    dst->GetContext().fpsr = ctx.fpsr;
}

void PhysicalCore::LogBacktrace(KernelCore& kernel) {
    auto* process = GetCurrentProcessPointer(kernel);
    if (!process) {
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->LogBacktrace(process);
    }
}

void PhysicalCore::Idle(KernelCore& kernel) {
    Common::Nx2Timeline::Span timeline_span{"guest core idle"};
    // EDEN_NX2_JIT_WARM=1: each emulated core has its own JIT, so code one core compiled is
    // compiled again by every other core the first time it runs it there (bursts of 60-180 ms
    // frames after area loads, 2026-11-10). An idle core compiles that code now, a few blocks at
    // a time, checking for an interrupt between blocks (fable52_answer_12). At most 20 ms of
    // every 100 ms per core: unbudgeted, the cores never rested while new code poured in at boot
    // and took the host CPU from the GPU thread and shader workers (first load 761 -> 811 ms,
    // hitch time 2.6 -> 3.7 s in the first 100 s; reloads 1.64 -> 1.34 s).
    if (Common::Nx2Hitch::JitWarmEnabled() && !m_is_interrupted.load(std::memory_order_acquire)) {
        if (KProcess* const process = kernel.ApplicationProcess()) {
            if (auto* const arm = process->GetArmInterface(m_core_index)) {
                using Clock = std::chrono::steady_clock;
                constexpr auto Window = std::chrono::milliseconds{100};
                constexpr auto Budget = std::chrono::milliseconds{20};
                thread_local std::vector<u64> keys;
                thread_local Clock::time_point window_start{};
                thread_local Clock::duration used{};
                auto now = Clock::now();
                if (now - window_start >= Window) {
                    window_start = now;
                    used = {};
                }
                while (used < Budget && !m_is_interrupted.load(std::memory_order_acquire) &&
                       Common::Nx2Hitch::TakeWarmKeys(static_cast<u32>(m_core_index), keys, 16) != 0) {
                    static_cast<void>(arm->WarmCompile(keys, m_is_interrupted));
                    const auto after = Clock::now();
                    used += after - now;
                    now = after;
                }
            }
        }
    }
    // Guest job systems hand work between cores many times per frame; blocking at once adds the
    // host's thread wake-up latency to every handoff. EDEN_NX2_IDLE_SPIN_US=<n>: poll for that
    // many microseconds before blocking. Switch 2 titles poll 100 us by default outside Android
    // (title 1's frames: 42 ms without, 35 ms with 50 us, 34 ms with 100 us, 36 ms with 300 us).
    static const long long spin_setting = [] {
        const char* text = std::getenv("EDEN_NX2_IDLE_SPIN_US");
        return text ? std::strtoll(text, nullptr, 10) : -1LL;
    }();
    long long spin_us = spin_setting;
    if (spin_us < 0) {
#ifdef __ANDROID__
        spin_us = 0;
#else
        const KProcess* process = kernel.ApplicationProcess();
        spin_us = process != nullptr && process->IsSwitch2() ? 100 : 0;
#endif
    }
    const auto spin = std::chrono::microseconds{spin_us};
    if (spin.count() > 0) {
        const auto deadline = std::chrono::steady_clock::now() + spin;
        if (IdlePauseOn()) {
            // EDEN_NX2_IDLE_PAUSE: the interrupt flag after every pause, the clock every 16.
            for (u32 n = 1; !m_is_interrupted.load(std::memory_order_acquire); ++n) {
                IdlePause();
                if ((n & 15) == 0 && std::chrono::steady_clock::now() >= deadline) {
                    break;
                }
            }
        } else {
            while (!m_is_interrupted.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
        }
    }
    if (m_core_lock) {
        // EDEN_NX2_CORE_LOCK: the wait compares the flag before sleeping, so a store made
        // after the load still wakes it (Interrupt notifies after every store).
        while (!m_is_interrupted.load(std::memory_order_acquire)) {
            m_is_interrupted.wait(false, std::memory_order_acquire);
        }
        return;
    }
    std::unique_lock lk{m_guard};
    m_on_interrupt.wait(lk, [this] { return m_is_interrupted.load(); });
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted;
}

void PhysicalCore::Interrupt() {
    if (m_core_lock) {
        {
            ScopedGuard lk{*this};
            m_is_interrupted.store(true, std::memory_order_seq_cst);
            if (m_arm_interface != nullptr) {
                m_arm_interface->SignalInterrupt(m_current_thread);
            }
        }
        // Wakes Idle if it sleeps; WaitOnAddress makes this cheap when nobody does.
        m_is_interrupted.notify_all();
        return;
    }

    // Lock core context.
    std::scoped_lock lk{m_guard};

    // Load members.
    auto* arm_interface = m_arm_interface;
    auto* thread = m_current_thread;

    // Add interrupt flag.
    m_is_interrupted = true;

    // Interrupt ourselves.
    m_on_interrupt.notify_one();

    // If there is no thread running, we are done.
    if (arm_interface == nullptr) {
        return;
    }

    // Interrupt the CPU.
    arm_interface->SignalInterrupt(thread);
}

void PhysicalCore::ClearInterrupt() {
    ScopedGuard lk{*this};
    m_is_interrupted = false;
}

} // namespace Kernel
