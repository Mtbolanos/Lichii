// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string>

namespace FrontendCommon {

std::filesystem::path ProfileImportStateDirectory();
bool NeedsLegacyProfileImport();
/// Fritchii v5.8.1, Help > Copy Data from Eden or Lichii: asks the next launch to copy a profile
/// over Fritchii's own (the current one kept as backups). `request`: the folder ("eden",
/// "lichii"...) on the first line, then what to copy (v5.8.2: saves=, settings=, mods=).
void RequestProfileImport(const std::string& request);
/// The request's text, or empty. Takes it (the request file is removed), puts back an
/// interrupted import and clears the done/fresh state, so the next import runs.
std::string TakeProfileImportRequest();
void CheckLegacyProfileBeforeConfig();
std::filesystem::path RemapImportedPath(const std::filesystem::path& value,
                                        const std::filesystem::path& source,
                                        const std::filesystem::path& destination);
void RemapImportedConfig(const std::filesystem::path& config_directory,
                         const std::filesystem::path& source,
                         const std::filesystem::path& destination);

} // namespace FrontendCommon
