// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/fs/profile_migration.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include "common/fs/file_lock.h"
#include "common/fs/symlink.h"

namespace Common::FS::ProfileMigration {
namespace {
namespace fs = std::filesystem;

struct Entry {
    Root root;
    fs::path stage;
    fs::path backup;
    fs::path source_backup;
    bool had_destination;
};

fs::path LockPath(const fs::path& state) {
    fs::create_directories(state);
    return state / "lock";
}

class MigrationLock {
public:
    explicit MigrationLock(const fs::path& state) : lock{LockPath(state)} {
        if (!lock.Held()) {
            throw std::runtime_error(
                "Another process is importing this profile, or its lock is unavailable");
        }
    }

private:
    Common::FS::FileLock lock;
};

bool Exists(const fs::path& path) {
    return fs::symlink_status(path).type() != fs::file_type::not_found;
}

void Touch(const fs::path& path) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file << "1\n";
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write migration state: " + path.string());
    }
}

bool Contains(const fs::path& parent, const fs::path& child) {
    auto p = parent.begin();
    auto c = child.begin();
    for (; p != parent.end() && c != child.end(); ++p, ++c) {
#ifdef _WIN32
        auto left = p->wstring();
        auto right = c->wstring();
        std::transform(left.begin(), left.end(), left.begin(), ::towlower);
        std::transform(right.begin(), right.end(), right.begin(), ::towlower);
        if (left != right) {
            return false;
        }
#else
        if (*p != *c) {
            return false;
        }
#endif
    }
    return p == parent.end();
}

bool Overlaps(const fs::path& left, const fs::path& right) {
    return Contains(left, right) || Contains(right, left);
}

fs::path Absolute(const fs::path& path) {
    auto result = fs::absolute(path).lexically_normal();
    while (result != result.root_path() && result.filename().empty()) {
        result = result.parent_path();
    }
    return result;
}

void CheckStateLocation(const Plan& plan) {
    if (plan.state_dir.empty()) {
        throw std::runtime_error("Migration state directory cannot be empty");
    }
    const auto state = fs::weakly_canonical(Absolute(plan.state_dir));
    for (const auto& root : plan.roots) {
        for (const auto& path : {root.source, root.destination}) {
            if (path.empty() || Overlaps(fs::weakly_canonical(Absolute(path)), state)) {
                throw std::runtime_error("Migration state must be outside every profile root");
            }
        }
    }
}

void CheckTree(const fs::path& root) {
    if (Common::FS::IsSymlink(root) || !fs::is_directory(root)) {
        throw std::runtime_error("Migration source must be a real directory: " + root.string());
    }
    for (const auto& item : fs::recursive_directory_iterator(root)) {
        const auto type = item.symlink_status().type();
        if (Common::FS::IsSymlink(item.path()) ||
            (type != fs::file_type::directory && type != fs::file_type::regular)) {
            throw std::runtime_error("Migration source contains a link or special file: " +
                                     item.path().string());
        }
    }
}

// A source root may itself be a link to a directory, for example a junction that keeps the old
// profile on another drive. Import from the directory it points to, so every overlap check and
// the recovery journal use the real path. Links inside the tree are still refused by CheckTree.
// Move is refused for a linked root: it would rename the real directory behind the link, which
// other links or programs may still use.
fs::path ResolveSourceRoot(const fs::path& root, Strategy strategy) {
    if (!Common::FS::IsSymlink(root)) {
        return root;
    }
    if (strategy == Strategy::Move) {
        throw std::runtime_error("Move cannot import a linked profile folder, use Copy or Link: " +
                                 root.string());
    }
    std::error_code error;
    const auto target = fs::canonical(root, error);
    if (error || Common::FS::IsSymlink(target) || !fs::is_directory(target)) {
        throw std::runtime_error("Migration source link does not lead to a real directory: " +
                                 root.string());
    }
    return target;
}

std::vector<Root> Validate(const Plan& plan) {
    if (plan.roots.empty() || plan.state_dir.empty()) {
        throw std::runtime_error("Migration requires roots and an external state directory");
    }
    const auto state = fs::weakly_canonical(Absolute(plan.state_dir));
    std::vector<Root> roots;
    std::vector<fs::path> paths;
    for (const auto& root : plan.roots) {
        if (root.source.empty() || root.destination.empty()) {
            throw std::runtime_error("Migration paths cannot be empty");
        }
        const auto source = ResolveSourceRoot(root.source, plan.strategy);
        CheckTree(source);
        const auto destination = Absolute(root.destination);
        Root normalized{fs::weakly_canonical(Absolute(source)),
                        fs::weakly_canonical(destination.parent_path()) / destination.filename()};
        if (Overlaps(normalized.source, fs::weakly_canonical(destination))) {
            throw std::runtime_error("Migration source and destination must not overlap");
        }
        for (const auto& path : {normalized.source, normalized.destination}) {
            if (path == path.root_path() || Overlaps(path, state)) {
                throw std::runtime_error("Migration state must be outside every profile root");
            }
            for (const auto& other : paths) {
                if (Overlaps(path, other)) {
                    throw std::runtime_error("Migration roots must not overlap");
                }
            }
            paths.push_back(path);
        }
        roots.push_back(std::move(normalized));
    }
    return roots;
}

void Verify(const fs::path& source, const fs::path& destination) {
    CheckTree(source);
    CheckTree(destination);
    std::size_t source_count{};
    const auto destination_count = static_cast<std::size_t>(std::distance(
        fs::recursive_directory_iterator(destination), fs::recursive_directory_iterator{}));
    for (const auto& item : fs::recursive_directory_iterator(source)) {
        ++source_count;
        const auto target = destination / item.path().lexically_relative(source);
        if (item.symlink_status().type() != fs::symlink_status(target).type()) {
            throw std::runtime_error("Migration verification found a different file type");
        }
        if (!item.is_regular_file()) {
            continue;
        }
        if (item.file_size() != fs::file_size(target)) {
            throw std::runtime_error("Migration verification found a different file size");
        }
        std::ifstream original{item.path(), std::ios::binary};
        std::ifstream copied{target, std::ios::binary};
        if (!original || !copied) {
            throw std::runtime_error("Cannot read files for migration verification");
        }
        std::array<char, 65536> a{}, b{};
        do {
            original.read(a.data(), a.size());
            copied.read(b.data(), b.size());
            if (original.gcount() != copied.gcount() ||
                !std::equal(a.begin(), a.begin() + original.gcount(), b.begin()) ||
                original.bad() || copied.bad()) {
                throw std::runtime_error("Migration verification found different file contents");
            }
        } while (original.gcount() != 0);
    }
    if (source_count != destination_count) {
        throw std::runtime_error("Migration verification found a different directory inventory");
    }
}

void Copy(const fs::path& source, const fs::path& stage) {
    fs::create_directory(stage);
    for (const auto& item : fs::recursive_directory_iterator(source)) {
        const auto target = stage / item.path().lexically_relative(source);
        if (item.symlink_status().type() == fs::file_type::directory) {
            fs::create_directory(target);
        } else if (item.symlink_status().type() == fs::file_type::regular) {
            fs::copy_file(item.path(), target);
        } else {
            throw std::runtime_error("Migration source changed to a link or special file");
        }
    }
    Verify(source, stage);
}

std::vector<Entry> ReadJournal(const fs::path& state) {
    std::ifstream file{state / "journal", std::ios::binary};
    std::string magic;
    std::size_t count{};
    file >> magic >> count;
    if (!file || magic != "lichii-profile-import-v1" || count == 0 || count > 1024) {
        throw std::runtime_error("Cannot read migration recovery journal");
    }
    std::vector<Entry> entries;
    for (std::size_t i = 0; i < count; ++i) {
        std::string source, destination, stage, backup, source_backup;
        bool had_destination{};
        file >> std::quoted(source) >> std::quoted(destination) >> std::quoted(stage) >>
            std::quoted(backup) >> std::quoted(source_backup) >> had_destination;
        if (!file) {
            throw std::runtime_error("Migration recovery journal is truncated");
        }
        const auto path = [](const std::string& text) {
            return fs::path{std::u8string(text.begin(), text.end())};
        };
        entries.push_back({{path(source), path(destination)},
                           path(stage),
                           path(backup),
                           path(source_backup),
                           had_destination});
        const auto& entry = entries.back();
        const auto prefix = entry.root.destination.native() + fs::path{".lichii-import-"}.native();
        const auto stage_name = entry.stage.native();
        auto expected_backup = entry.stage;
        expected_backup.replace_extension(".backup");
        if (!entry.root.source.is_absolute() || !entry.root.destination.is_absolute() ||
            !stage_name.starts_with(prefix) || !stage_name.ends_with(fs::path{".stage"}.native()) ||
            entry.stage.parent_path() != entry.root.destination.parent_path() ||
            entry.backup != expected_backup ||
            Overlaps(entry.root.source, entry.root.destination) ||
            Overlaps(entry.root.source, Absolute(state)) ||
            Overlaps(entry.root.destination, Absolute(state))) {
            throw std::runtime_error("Migration recovery journal contains unsafe paths");
        }
        auto expected_source_backup = entry.root.source;
        expected_source_backup +=
            entry.backup.native().substr(entry.root.destination.native().size());
        if (entry.source_backup != expected_source_backup) {
            throw std::runtime_error("Migration recovery journal contains an unsafe source backup");
        }
    }
    return entries;
}

void WriteJournal(const fs::path& state, const std::vector<Entry>& entries) {
    std::ofstream file{state / "journal.tmp", std::ios::binary | std::ios::trunc};
    file << "lichii-profile-import-v1 " << entries.size() << '\n';
    for (const auto& entry : entries) {
        for (const auto& path : {entry.root.source, entry.root.destination, entry.stage,
                                 entry.backup, entry.source_backup}) {
            const auto text = path.u8string();
            file << std::quoted(std::string(text.begin(), text.end())) << ' ';
        }
        file << entry.had_destination << '\n';
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write migration recovery journal");
    }
    fs::rename(state / "journal.tmp", state / "journal");
}

fs::path Marker(const fs::path& state, const std::string& name, std::size_t index) {
    return state / (name + std::to_string(index));
}

void RemoveMarkers(const fs::path& state) {
    for (const auto& item : fs::directory_iterator(state)) {
        const auto name = item.path().filename().string();
        if (name.starts_with("publishing-") || name.starts_with("recovered-")) {
            fs::remove(item.path());
        }
    }
}

} // namespace

State GetState(const fs::path& state_dir) {
    if (Exists(state_dir / "complete")) {
        return State::Complete;
    }
    if (Exists(state_dir / "pending") || Exists(state_dir / "journal")) {
        return State::Pending;
    }
    return Exists(state_dir / "fresh") ? State::Fresh : State::Unresolved;
}

bool NeedsMigration(const fs::path& state_dir) {
    const auto state = GetState(state_dir);
    return state == State::Unresolved || state == State::Pending;
}

void MarkFresh(const fs::path& state_dir) {
    const MigrationLock lock{state_dir};
    if (GetState(state_dir) == State::Pending) {
        throw std::runtime_error("Recover the pending import before starting fresh");
    }
    fs::create_directories(state_dir);
    Touch(state_dir / "fresh");
}

namespace {

void RecoverUnlocked(const fs::path& state_dir) {
    if (GetState(state_dir) != State::Pending) {
        return;
    }
    if (!Exists(state_dir / "journal")) {
        RemoveMarkers(state_dir);
        fs::remove(state_dir / "pending");
        fs::remove(state_dir / "journal.tmp");
        return;
    }
    const auto entries = ReadJournal(state_dir);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (Exists(Marker(state_dir, "recovered-", i))) {
            continue;
        }
        if (Exists(entry.source_backup)) {
            if (Exists(entry.root.source)) {
                throw std::runtime_error("Both source and its migration backup exist; keep both");
            }
            fs::rename(entry.source_backup, entry.root.source);
        }
        if (Exists(Marker(state_dir, "publishing-", i)) && !Exists(entry.stage) &&
            Exists(entry.root.destination)) {
            fs::rename(entry.root.destination, entry.stage);
        }
        if (Exists(entry.backup)) {
            if (Exists(entry.root.destination)) {
                throw std::runtime_error(
                    "Both destination and its migration backup exist; keep both");
            }
            fs::rename(entry.backup, entry.root.destination);
        }
        Touch(Marker(state_dir, "recovered-", i));
    }
    for (const auto& entry : entries) {
        if (Common::FS::IsSymlink(entry.stage)) {
            fs::remove(entry.stage);
        } else {
            fs::remove_all(entry.stage);
        }
    }
    fs::remove(state_dir / "journal");
    RemoveMarkers(state_dir);
    fs::remove(state_dir / "pending");
}

} // namespace

void Recover(const fs::path& state_dir) {
    const MigrationLock lock{state_dir};
    RecoverUnlocked(state_dir);
}

void RunImport(const Plan& plan, const Prepare& prepare) {
    CheckStateLocation(plan);
    const MigrationLock lock{plan.state_dir};
    if (!NeedsMigration(plan.state_dir)) {
        return;
    }
    RecoverUnlocked(plan.state_dir);
    const auto roots = Validate(plan);
    fs::create_directories(plan.state_dir);
    Touch(plan.state_dir / "pending");
    const auto id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::vector<Entry> entries;
    for (const auto& root : roots) {
        auto stage = root.destination;
        stage += ".lichii-import-" + id + ".stage";
        auto backup = root.destination;
        backup += ".lichii-import-" + id + ".backup";
        auto source_backup = root.source;
        source_backup += ".lichii-import-" + id + ".backup";
        if (Exists(stage) || Exists(backup) || Exists(source_backup)) {
            throw std::runtime_error("Migration backup paths already exist");
        }
        entries.push_back({root, stage, backup, source_backup, Exists(root.destination)});
    }
    WriteJournal(plan.state_dir, entries);
    std::vector<Root> staged;
    try {
        for (const auto& entry : entries) {
            fs::create_directories(entry.stage.parent_path());
            if (plan.strategy == Strategy::Link) {
                const bool linked = plan.create_link
                                        ? plan.create_link(entry.root.source, entry.stage)
                                        : Common::FS::CreateSymlink(entry.root.source, entry.stage);
                if (!linked || !Common::FS::IsSymlink(entry.stage) ||
                    !fs::equivalent(entry.root.source, entry.stage)) {
                    throw std::runtime_error("Could not link migration source: " +
                                             entry.root.source.string());
                }
            } else {
                Copy(entry.root.source, entry.stage);
            }
            staged.push_back({entry.root.source, entry.stage});
        }
        if (prepare && plan.strategy != Strategy::Link) {
            prepare(staged);
        }
        for (const auto& entry : entries) {
            if (!Exists(entry.stage)) {
                throw std::runtime_error("Prepared migration root is missing");
            }
        }
    } catch (...) {
        RecoverUnlocked(plan.state_dir);
        Touch(plan.state_dir / "pending");
        throw;
    }
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        Touch(Marker(plan.state_dir, "publishing-", i));
        if (entry.had_destination) {
            fs::rename(entry.root.destination, entry.backup);
        } else if (Exists(entry.root.destination)) {
            throw std::runtime_error("Migration destination appeared during staging");
        }
        fs::rename(entry.stage, entry.root.destination);
    }
    if (plan.strategy == Strategy::Move) {
        for (const auto& entry : entries) {
            fs::rename(entry.root.source, entry.source_backup);
        }
    }
    Touch(plan.state_dir / "complete");
    fs::remove(plan.state_dir / "pending");
}

} // namespace Common::FS::ProfileMigration
