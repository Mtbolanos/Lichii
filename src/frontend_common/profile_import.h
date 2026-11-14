// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>

namespace FrontendCommon {

std::filesystem::path ProfileImportStateDirectory();
bool NeedsLegacyProfileImport();
void CheckLegacyProfileBeforeConfig();
std::filesystem::path RemapImportedPath(const std::filesystem::path& value,
                                        const std::filesystem::path& source,
                                        const std::filesystem::path& destination);
void RemapImportedConfig(const std::filesystem::path& config_directory,
                         const std::filesystem::path& source,
                         const std::filesystem::path& destination);

} // namespace FrontendCommon
