// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <QFutureWatcher>
#include <QImage>
#include <QObject>
#include <QString>

#include "common/common_types.h"

class QWidget;

enum class IconSearchStatus {
    Found,
    Choices,
    NotFound,
    Timeout,
    NetworkError,
    Cancelled,
    NeedsApiKey,
    BadApiKey,
    InternalError,
};

struct IconSearchResult {
    IconSearchStatus status{IconSearchStatus::NotFound};
    std::vector<u8> icon_bytes;
    std::vector<std::vector<u8>> choices;
    std::string resolved_title;
    std::string error;
};

enum class ArtworkLookupStatus { Found, NotFound, BadApiKey, NetworkError, Cancelled };

struct ArtworkLookupResult {
    ArtworkLookupStatus status{ArtworkLookupStatus::NotFound};
    QImage image;
    std::string detail; // Never contains the API key.
};

// Finds square artwork for a game title on SteamGridDB with the user's API key: the best title
// match, then its first 512x512 or 1024x1024 grid, else its first icon. A year at the end of the
// title, like "(2026)", picks the entry released that year. prefer_newest picks the newest entry
// of the best match's name (remakes share the name of the original). Blocking, so run it off the
// UI thread; it stops within 30 seconds or as soon as cancel is set.
ArtworkLookupResult LookUpSteamGridDbArtwork(const std::string& title, const std::string& api_key,
                                             const std::atomic_bool& cancel,
                                             bool prefer_newest = false);

// Looks an icon up by game name on switchiconshowdown.com, falling back to SteamGridDB when the
// user has set an API key. The search runs in the background under one 10 second deadline; errors,
// the API key prompt and the choice between several icons are handled here, so a caller only
// connects IconSelected.
class IconSearch : public QObject {
    Q_OBJECT
public:
    explicit IconSearch(QWidget* parent);
    ~IconSearch() override;

    void Start(const QString& name, u64 title_id = 0);
    bool IsRunning() const;

signals:
    void RunningChanged(bool running);
    // A 256x256 PNG, and the canonical title when the name was an alias (else empty).
    void IconSelected(const QByteArray& png, const QString& resolved_title);

private:
    void OnFinished();
    int ChooseIcon(const std::vector<std::vector<u8>>& choices);
    bool AskForApiKey(bool missing);

    QWidget* parent_widget;
    QFutureWatcher<IconSearchResult> watcher;
    std::shared_ptr<std::atomic_bool> cancel_flag{std::make_shared<std::atomic_bool>(false)};
    QString last_name;
    u64 last_title_id{};
};
