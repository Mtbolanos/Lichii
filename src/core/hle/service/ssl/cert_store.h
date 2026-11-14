// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/hle/result.h"
#include "core/hle/service/ssl/ssl_types.h"

namespace Core {
class System;
}

namespace Service::SSL {

class CertStore {
public:
    explicit CertStore(Core::System& system);
    ~CertStore();

    Result GetCertificates(u32* out_num_entries, std::span<u8> out_data,
                           std::span<const CaCertificateId> certificate_ids);
    Result GetCertificateBufSize(u32* out_size, u32* out_num_entries,
                                 std::span<const CaCertificateId> certificate_ids);

private:
    template <typename F>
    void ForEachCertificate(std::span<const CaCertificateId> certs, F&& f);

private:
    struct Certificate {
        TrustedCertStatus status;
        std::vector<u8> der_data;
    };

    std::map<CaCertificateId, Certificate> m_certs;
    /// The host redirect's CA (online_redirect_ca), under an id no system certificate uses. It
    /// is in every list the store hands out: titles that run TLS in-process build their trust
    /// store from these lists, whatever ids they ask for.
    std::optional<std::pair<const CaCertificateId, Certificate>> m_test_ca;
};

} // namespace Service::SSL
