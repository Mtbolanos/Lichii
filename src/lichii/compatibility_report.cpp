// SPDX-License-Identifier: GPL-3.0-or-later

#include "compatibility_report.h"

#include <cmath>
#include <memory>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QTabWidget>
#include <QTimer>
#include <QUrlQuery>
#include <QUuid>
#include <QVBoxLayout>

namespace CompatibilityReporting {
namespace {
constexpr auto Site = "https://lichii.zortos.me";
constexpr qsizetype ReplyLimit = 256 * 1024;
const QMap<QString, int> Limits{
    {QStringLiteral("name"), 60},      {QStringLiteral("handle"), 16},
    {QStringLiteral("build"), 100},    {QStringLiteral("gameVersion"), 60},
    {QStringLiteral("cpu"), 160},      {QStringLiteral("gpu"), 160},
    {QStringLiteral("driver"), 160},   {QStringLiteral("os"), 160},
    {QStringLiteral("ram"), 60},       {QStringLiteral("settings"), 3000},
    {QStringLiteral("progress"), 500}, {QStringLiteral("summary"), 120},
    {QStringLiteral("details"), 6000}, {QStringLiteral("reproduction"), 3000},
    {QStringLiteral("updates"), 500},  {QStringLiteral("mods"), 500},
};
const QStringList OptionalFields{QStringLiteral("handle"), QStringLiteral("ram"),
                                 QStringLiteral("reproduction"), QStringLiteral("updates"),
                                 QStringLiteral("mods")};

QString Known(const QString& value) {
    return value.trimmed().isEmpty() ? QStringLiteral("Unknown") : value.trimmed();
}

bool HasControls(const QString& value) {
    for (const auto ch : value) {
        if (ch.category() == QChar::Other_Control && ch != QLatin1Char('\t') &&
            ch != QLatin1Char('\n') && ch != QLatin1Char('\r')) {
            return true;
        }
    }
    return false;
}
} // namespace

QString SummarizeEnabledAddons(const QList<Addon>& addons) {
    QStringList labels;
    for (const auto& addon : addons) {
        if (!addon.enabled || addon.name.trimmed().isEmpty()) {
            continue;
        }
        auto label = addon.name.trimmed();
        if (!addon.version.trimmed().isEmpty()) {
            label += QStringLiteral(" (%1)").arg(addon.version.trimmed());
        }
        if (!labels.contains(label)) {
            labels.append(std::move(label));
        }
    }
    if (labels.isEmpty()) {
        return QStringLiteral("None listed as enabled at launch.");
    }
    const QString summary =
        QStringLiteral("Enabled at launch: ") + labels.join(QStringLiteral("; "));
    constexpr int limit = 500;
    if (summary.size() <= limit) {
        return summary;
    }
    const auto suffix = QStringLiteral("… (list truncated; review)");
    auto shortened = summary.left(limit - suffix.size());
    if (!shortened.isEmpty() && shortened.back().isHighSurrogate()) {
        shortened.chop(1);
    }
    return shortened + suffix;
}

Dialog::Dialog(const Session& session, QWidget* parent) : Dialog(session, nullptr, parent) {}

Dialog::Dialog(const Session& session, QNetworkAccessManager* network_, QWidget* parent)
    : QDialog(parent), network(network_ ? network_ : new QNetworkAccessManager(this)) {
    setWindowTitle(tr("Submit compatibility report"));
    resize(760, 790);
    auto* layout = new QVBoxLayout(this);
    auto* introduction = new QLabel(
        tr("Share your experience with the latest completed session. Nothing is sent until you "
           "choose Search or Submit. Logs are not uploaded. No account is required."),
        this);
    introduction->setWordWrap(true);
    layout->addWidget(introduction);
    auto* search_row = new QHBoxLayout;
    query = new QLineEdit(session.title, this);
    query->setObjectName(QStringLiteral("catalogQuery"));
    query->setMaxLength(150);
    query->setPlaceholderText(tr("Game title to search"));
    search = new QPushButton(tr("Search catalog"), this);
    search->setObjectName(QStringLiteral("search"));
    search_row->addWidget(query, 1);
    search_row->addWidget(search);
    layout->addLayout(search_row);
    auto* search_notice = new QLabel(tr("Search sends only the text above to lichii.zortos.me. "
                                        "The service sees your IP address."),
                                     this);
    search_notice->setWordWrap(true);
    layout->addWidget(search_notice);
    auto* results_row = new QHBoxLayout;
    games = new QComboBox(this);
    games->setObjectName(QStringLiteral("catalogResults"));
    previous = new QPushButton(tr("Previous"), this);
    next = new QPushButton(tr("Next"), this);
    previous->setObjectName(QStringLiteral("previous"));
    next->setObjectName(QStringLiteral("next"));
    confirm = new QPushButton(tr("Confirm game"), this);
    confirm->setObjectName(QStringLiteral("confirm"));
    results_row->addWidget(games, 1);
    results_row->addWidget(previous);
    results_row->addWidget(next);
    results_row->addWidget(confirm);
    layout->addLayout(results_row);
    match = new QLabel(tr("No catalog game confirmed."), this);
    match->setTextFormat(Qt::PlainText);
    match->setWordWrap(true);
    layout->addWidget(match);

    tabs = new QTabWidget(this);
    layout->addWidget(tabs, 1);
    const auto add_page = [this](const QString& title) {
        auto* content = new QWidget;
        auto* form = new QFormLayout(content);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        auto* scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setWidget(content);
        tabs->addTab(scroll, title);
        return form;
    };
    const auto line = [this](QFormLayout* form, const QString& key, const QString& label,
                             const QString& value = QString{}) {
        auto* edit = new QLineEdit(value, this);
        edit->setObjectName(key);
        edit->setMaxLength(Limits.value(key));
        fields.insert(key, edit);
        form->addRow(label, edit);
    };
    const auto paragraph = [this](QFormLayout* form, const QString& key, const QString& label,
                                  const QString& value = QString{}) {
        auto* edit = new QPlainTextEdit(value, this);
        edit->setObjectName(key);
        edit->setMinimumHeight(90);
        paragraphs.insert(key, edit);
        form->addRow(label, edit);
    };
    auto* experience = add_page(tr("Experience"));
    line(experience, QStringLiteral("name"), tr("Public nickname"), QStringLiteral("Anonymous"));
    line(experience, QStringLiteral("handle"), tr("X handle (optional)"));
    outcome = new QComboBox(this);
    outcome->setObjectName(QStringLiteral("outcome"));
    outcome->addItem(tr("Choose your assessment"), QString{});
    for (const auto& value : {"Perfect", "Playable", "Runs", "Loads", "Unplayable"}) {
        outcome->addItem(QString::fromLatin1(value), QString::fromLatin1(value));
    }
    experience->addRow(tr("Compatibility"), outcome);
    line(experience, QStringLiteral("summary"), tr("Summary"));
    paragraph(experience, QStringLiteral("progress"), tr("How far did you play?"));
    paragraph(experience, QStringLiteral("details"), tr("What worked or failed?"));
    paragraph(experience, QStringLiteral("reproduction"), tr("Steps to reproduce (optional)"));
    auto* setup = add_page(tr("System and settings"));
    auto* setup_notice =
        new QLabel(tr("Prefilled values describe launch settings, not measured "
                      "performance. Review changes made during play. Unknown means "
                      "not collected. Do not paste logs, paths, keys or private details."));
    setup_notice->setWordWrap(true);
    setup->addRow(setup_notice);
    line(setup, QStringLiteral("build"), tr("Lichii build"), Known(session.build));
    line(setup, QStringLiteral("gameVersion"), tr("Game version"), Known(session.game_version));
    line(setup, QStringLiteral("cpu"), tr("CPU"), Known(session.cpu));
    line(setup, QStringLiteral("gpu"), tr("GPU"), Known(session.gpu));
    line(setup, QStringLiteral("driver"), tr("GPU driver / OpenGL version"), Known(session.driver));
    line(setup, QStringLiteral("os"), tr("Operating system"), Known(session.os));
    line(setup, QStringLiteral("ram"), tr("RAM (optional)"), session.ram);
    graphics_api = new QComboBox(this);
    graphics_api->setObjectName(QStringLiteral("graphicsApi"));
    graphics_api->addItems(
        {QStringLiteral("Unknown"), QStringLiteral("Vulkan"), QStringLiteral("OpenGL")});
    graphics_api->setCurrentText(Known(session.graphics_api));
    setup->addRow(tr("Graphics API"), graphics_api);
    paragraph(setup, QStringLiteral("settings"), tr("Launch settings"), Known(session.settings));
    paragraph(setup, QStringLiteral("updates"), tr("Updates (optional)"), session.updates);
    paragraph(setup, QStringLiteral("mods"), tr("Mods (optional)"), session.mods);
    preview = new QPlainTextEdit(this);
    preview->setObjectName(QStringLiteral("preview"));
    preview->setReadOnly(true);
    tabs->addTab(preview, tr("Public preview"));

    auto* privacy = new QLabel(tr("All report fields in the preview will be public. Logs, files, "
                                  "local title IDs and play history are not sent. The server uses "
                                  "your IP address for rate limiting. Review and clear anything "
                                  "you do not want to share."),
                               this);
    privacy->setWordWrap(true);
    layout->addWidget(privacy);
    public_consent = new QCheckBox(tr("I understand my name, optional X handle, report text, "
                                      "hardware and settings will be public."),
                                   this);
    safe_files_consent = new QCheckBox(tr("My report contains no game files, keys, firmware, "
                                          "saves or private account details."),
                                       this);
    public_consent->setObjectName(QStringLiteral("publicConsent"));
    safe_files_consent->setObjectName(QStringLiteral("safeFilesConsent"));
    layout->addWidget(public_consent);
    layout->addWidget(safe_files_consent);
    status = new QLabel(this);
    status->setObjectName(QStringLiteral("status"));
    status->setTextFormat(Qt::PlainText);
    status->setWordWrap(true);
    layout->addWidget(status);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel, this);
    clear = buttons->addButton(tr("Clear draft"), QDialogButtonBox::ResetRole);
    submit = buttons->addButton(tr("Submit report"), QDialogButtonBox::ActionRole);
    open_receipt = buttons->addButton(tr("Open public report"), QDialogButtonBox::ActionRole);
    clear->setObjectName(QStringLiteral("clear"));
    submit->setObjectName(QStringLiteral("submit"));
    open_receipt->setVisible(false);
    layout->addWidget(buttons);
    for (auto* button : findChildren<QPushButton*>()) {
        button->setAutoDefault(false);
        button->setDefault(false);
    }
    connect(buttons, &QDialogButtonBox::rejected, this, &Dialog::reject);
    connect(search, &QPushButton::clicked, this, [this] {
        page = 1;
        Search();
    });
    connect(previous, &QPushButton::clicked, this, [this] {
        --page;
        Search();
    });
    connect(next, &QPushButton::clicked, this, [this] {
        ++page;
        Search();
    });
    connect(confirm, &QPushButton::clicked, this, &Dialog::ConfirmGame);
    connect(submit, &QPushButton::clicked, this, &Dialog::Submit);
    connect(clear, &QPushButton::clicked, this, &Dialog::Clear);
    connect(open_receipt, &QPushButton::clicked, this, [this] {
        if (!receipt_url.isEmpty()) {
            QDesktopServices::openUrl(receipt_url);
        }
    });
    connect(query, &QLineEdit::textChanged, this, [this] {
        Abort();
        games->clear();
        confirmed_game = 0;
        page = page_count = 1;
        match->setText(tr("No catalog game confirmed."));
        Edited();
    });
    connect(games, &QComboBox::currentIndexChanged, this, [this] {
        confirmed_game = 0;
        match->setText(tr("No catalog game confirmed."));
        Edited();
    });
    for (auto* field : fields) {
        connect(field, &QLineEdit::textChanged, this, &Dialog::Edited);
    }
    for (auto* field : paragraphs) {
        connect(field, &QPlainTextEdit::textChanged, this, &Dialog::Edited);
    }
    connect(outcome, &QComboBox::currentIndexChanged, this, &Dialog::Edited);
    connect(graphics_api, &QComboBox::currentIndexChanged, this, &Dialog::Edited);
    connect(public_consent, &QCheckBox::toggled, this, &Dialog::Refresh);
    connect(safe_files_consent, &QCheckBox::toggled, this, &Dialog::Refresh);
    auto* cooldown = new QTimer(this);
    connect(cooldown, &QTimer::timeout, this, &Dialog::Refresh);
    cooldown->start(1000);
    Refresh();
}

Dialog::~Dialog() {
    state = State::Closed;
    Abort();
}

void Dialog::reject() {
    state = State::Closed;
    Abort();
    QDialog::reject();
}

void Dialog::Abort() {
    if (reply) {
        auto* pending = reply.data();
        reply = nullptr;
        pending->abort();
        pending->deleteLater();
    }
    if (state == State::Searching) {
        state = State::Editing;
    }
}

void Dialog::Edited() {
    public_consent->setChecked(false);
    safe_files_consent->setChecked(false);
    submitted_payload.clear();
    submission_key.clear();
    Refresh();
}

void Dialog::Clear() {
    if (state != State::Editing && state != State::Searching) {
        return;
    }
    Abort();
    query->clear();
    games->clear();
    confirmed_game = 0;
    for (auto* field : fields) {
        field->clear();
    }
    for (auto* field : paragraphs) {
        field->clear();
    }
    outcome->setCurrentIndex(0);
    graphics_api->setCurrentIndex(0);
    Edited();
    status->setText(tr("Draft cleared. Nothing was sent by clearing it."));
}

QByteArray Dialog::Payload() const {
    QJsonObject object;
    for (auto it = fields.cbegin(); it != fields.cend(); ++it) {
        object.insert(it.key(), it.value()->text().trimmed());
    }
    for (auto it = paragraphs.cbegin(); it != paragraphs.cend(); ++it) {
        object.insert(it.key(), it.value()->toPlainText().trimmed());
    }
    object.insert(QStringLiteral("gameId"), confirmed_game);
    object.insert(QStringLiteral("graphicsApi"), graphics_api->currentText());
    object.insert(QStringLiteral("outcome"), outcome->currentData().toString());
    object.insert(QStringLiteral("publicConsent"), public_consent->isChecked());
    object.insert(QStringLiteral("safeFilesConsent"), safe_files_consent->isChecked());
    return QJsonDocument(object).toJson(QJsonDocument::Compact);
}

QString Dialog::ValidationError() const {
    if (confirmed_game == 0) {
        return tr("Search for and confirm the catalog game first.");
    }
    const auto object = QJsonDocument::fromJson(Payload()).object();
    for (auto it = Limits.cbegin(); it != Limits.cend(); ++it) {
        const auto value = object.value(it.key()).toString();
        if ((!OptionalFields.contains(it.key()) && value.isEmpty()) || value.size() > it.value() ||
            HasControls(value)) {
            return tr("Check %1: required fields must be filled, within their length limit, "
                      "and contain no control characters.")
                .arg(it.key());
        }
    }
    static const QRegularExpression handle_pattern(QStringLiteral("^@?[A-Za-z0-9_]{1,15}$"));
    const auto handle = object.value(QStringLiteral("handle")).toString();
    if (!handle.isEmpty() && !handle_pattern.match(handle).hasMatch()) {
        return tr(
            "An X handle must contain 1–15 letters, digits or underscores, with an optional @.");
    }
    if (outcome->currentData().toString().isEmpty()) {
        return tr("Choose your own compatibility assessment.");
    }
    if (Payload().size() > 128 * 1024) {
        return tr("The report is too large.");
    }
    return {};
}

void Dialog::Refresh() {
    const bool editing = state == State::Editing;
    const bool searching = state == State::Searching;
    const bool cooling_down = QDateTime::currentMSecsSinceEpoch() < retry_at;
    query->setEnabled(editing || searching);
    search->setEnabled(editing && !cooling_down);
    games->setEnabled(editing);
    confirm->setEnabled(editing && games->currentIndex() >= 0);
    previous->setEnabled(editing && page > 1 && !cooling_down);
    next->setEnabled(editing && page < page_count && !cooling_down);
    clear->setEnabled(editing || searching);
    tabs->setEnabled(editing || searching || state == State::Finished);
    public_consent->setEnabled(editing);
    safe_files_consent->setEnabled(editing);
    const auto validation = ValidationError();
    submit->setEnabled(editing && !cooling_down && validation.isEmpty() &&
                       public_consent->isChecked() && safe_files_consent->isChecked());
    submit->setToolTip(validation);
    const auto content =
        QString::fromUtf8(QJsonDocument::fromJson(Payload()).toJson(QJsonDocument::Indented));
    if (preview->toPlainText() != content) {
        preview->setPlainText(content);
    }
}

void Dialog::Search() {
    if (state != State::Editing || QDateTime::currentMSecsSinceEpoch() < retry_at) {
        return;
    }
    if (HasControls(query->text()) || query->text().trimmed().size() > 150) {
        status->setText(tr("Check the search text."));
        return;
    }
    confirmed_game = 0;
    games->clear();
    Edited();
    state = State::Searching;
    status->setText(tr("Searching catalog… Cancel closes the dialog and stops this request."));
    Refresh();
    Request(false);
}

void Dialog::ConfirmGame() {
    if (state != State::Editing || games->currentIndex() < 0) {
        return;
    }
    if (confirmed_game == games->currentData().toLongLong()) {
        return;
    }
    confirmed_game = games->currentData().toLongLong();
    confirmed_title = games->currentText();
    match->setText(tr("Confirmed: %1 (catalog ID %2)").arg(confirmed_title).arg(confirmed_game));
    Edited();
}

void Dialog::Submit() {
    if (state != State::Editing || QDateTime::currentMSecsSinceEpoch() < retry_at ||
        !public_consent->isChecked() || !safe_files_consent->isChecked()) {
        return;
    }
    const auto error = ValidationError();
    if (!error.isEmpty()) {
        status->setText(error);
        return;
    }
    const auto payload = Payload();
    if (submitted_payload != payload || submission_key.isEmpty()) {
        submitted_payload = payload;
        submission_key = QUuid::createUuid().toString(QUuid::WithoutBraces).toUtf8();
    }
    state = State::Submitting;
    status->setText(tr("Submitting… Closing stops waiting, but cannot retract a report the "
                       "server already received. There are no automatic retries."));
    Refresh();
    Request(true, submitted_payload);
}

void Dialog::Request(bool submission, const QByteArray& payload) {
    QUrl url(QString::fromLatin1(Site) + (submission ? QStringLiteral("/api/v1/desktop/reports")
                                                     : QStringLiteral("/api/games")));
    if (!submission) {
        QUrlQuery params;
        params.addQueryItem(QStringLiteral("q"),
                            QString::fromLatin1(QUrl::toPercentEncoding(query->text().trimmed())));
        params.addQueryItem(QStringLiteral("page"), QString::number(page));
        params.addQueryItem(QStringLiteral("sort"), QStringLiteral("title"));
        url.setQuery(params);
    }
    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::ManualRedirectPolicy);
    request.setAttribute(QNetworkRequest::CookieLoadControlAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::CookieSaveControlAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::AuthenticationReuseAttribute, QNetworkRequest::Manual);
    request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                         QNetworkRequest::AlwaysNetwork);
    request.setAttribute(QNetworkRequest::CacheSaveControlAttribute, false);
    request.setTransferTimeout(20000);
    request.setRawHeader("Accept", "application/json");
    if (submission) {
        request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
        request.setRawHeader("X-Lichii-Client", "desktop/1");
        request.setRawHeader("Idempotency-Key", submission_key);
    }
    auto* pending = submission ? network->post(request, payload) : network->get(request);
    reply = pending;
    QTimer::singleShot(20000, this, [this, pending = QPointer<QNetworkReply>(pending)] {
        if (pending && reply == pending) {
            pending->abort();
        }
    });
    pending->setReadBufferSize(ReplyLimit + 1);
    auto body = std::make_shared<QByteArray>();
    connect(pending, &QNetworkReply::readyRead, this, [this, pending, body] {
        if (reply != pending) {
            return;
        }
        body->append(pending->read(ReplyLimit + 1 - body->size()));
        if (body->size() > ReplyLimit) {
            pending->abort();
        }
    });
    connect(pending, &QNetworkReply::finished, this, [this, pending, submission, body] {
        pending->deleteLater();
        if (reply != pending || state == State::Closed) {
            return;
        }
        reply = nullptr;
        body->append(pending->read(ReplyLimit + 1 - body->size()));
        Finished(submission, pending->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt(),
                 *body, pending->rawHeader("Retry-After"),
                 pending->error() != QNetworkReply::NoError || body->size() > ReplyLimit);
    });
}

void Dialog::Finished(bool submission, int http_status, const QByteArray& body,
                      const QByteArray& retry_after, bool network_error) {
    state = State::Editing;
    if (http_status == 429) {
        bool valid = false;
        const auto seconds = retry_after.toLongLong(&valid);
        const auto delay = valid && seconds >= 0 && seconds <= 31536000 ? seconds : 3600;
        retry_at = QDateTime::currentMSecsSinceEpoch() + delay * 1000;
        status->setText(tr("Rate limit reached. Wait %1 seconds, then retry explicitly. "
                           "Nothing will be retried automatically.")
                            .arg(delay));
    } else if (http_status == 400) {
        const auto errors =
            QJsonDocument::fromJson(body).object().value(QStringLiteral("fields")).toObject();
        QStringList invalid_fields;
        for (auto it = errors.constBegin(); it != errors.constEnd(); ++it) {
            if (Limits.contains(it.key()) || it.key() == QStringLiteral("gameId") ||
                it.key() == QStringLiteral("outcome") ||
                it.key() == QStringLiteral("graphicsApi") ||
                it.key() == QStringLiteral("publicConsent") ||
                it.key() == QStringLiteral("safeFilesConsent")) {
                invalid_fields.append(it.key());
            }
        }
        status->setText(invalid_fields.isEmpty()
                            ? tr("The server rejected the report. Review the fields and length "
                                 "limits before submitting again.")
                            : tr("The server rejected these fields: %1. Review them before "
                                 "submitting again.")
                                  .arg(invalid_fields.join(QStringLiteral(", "))));
    } else if (http_status == 409) {
        status->setText(
            tr("The server rejected this submission identity. Edit the draft to start "
               "a new submission; first check whether your report was already accepted."));
    } else if (http_status == 403 || http_status == 413 || http_status == 415) {
        status->setText(
            tr("The server rejected the request (HTTP %1). No automatic retry was made.")
                .arg(http_status));
    } else if (network_error || (http_status != 200 && !(submission && http_status == 201))) {
        status->setText(
            submission ? tr("No valid receipt was received. Delivery may have succeeded. "
                            "An explicit retry of this unchanged draft uses the same submission "
                            "identity. Closing loses that identity; do not create a duplicate.")
                       : tr("Catalog search failed. Check your connection and retry explicitly."));
    } else {
        QJsonParseError parse_error;
        const auto document = QJsonDocument::fromJson(body, &parse_error);
        const auto object = document.object();
        bool valid = parse_error.error == QJsonParseError::NoError && document.isObject();
        if (submission) {
            const auto id = object.value(QStringLiteral("id")).toString();
            const QUrl url(object.value(QStringLiteral("url")).toString());
            const auto uuid = QUuid(id);
            valid = valid && !uuid.isNull() && uuid.toString(QUuid::WithoutBraces) == id &&
                    url.isValid() && url.scheme() == QStringLiteral("https") &&
                    url.host() == QStringLiteral("lichii.zortos.me") && url.port(-1) == -1 &&
                    url.userInfo().isEmpty() && !url.hasQuery() && !url.hasFragment() &&
                    url.path() == QStringLiteral("/reports/") + id;
            if (valid) {
                state = State::Finished;
                receipt_url = url;
                open_receipt->setVisible(true);
                status->setText(tr("Report submitted. It is public. Thank you."));
            } else {
                status->setText(tr("The server returned an invalid receipt. The report may have "
                                   "been accepted. Only retry this unchanged draft explicitly."));
            }
        } else {
            const auto entries = object.value(QStringLiteral("games")).toArray();
            const auto returned_page = object.value(QStringLiteral("page")).toInt(-1);
            const auto pages = object.value(QStringLiteral("pageCount")).toInt(-1);
            valid = valid && object.value(QStringLiteral("games")).isArray() &&
                    entries.size() <= 15 && returned_page == page && pages >= 1 && pages <= 10000;
            for (const auto& entry : entries) {
                const auto game = entry.toObject();
                const double id = game.value(QStringLiteral("sgdbId")).toDouble(-1);
                const auto title = game.value(QStringLiteral("title")).toString();
                valid = valid && std::isfinite(id) && id > 0 && id <= 9007199254740991.0 &&
                        std::floor(id) == id && !title.trimmed().isEmpty() &&
                        title.size() <= 1000 && !HasControls(title) &&
                        game.value(QStringLiteral("slug")).isString();
            }
            if (valid) {
                for (const auto& entry : entries) {
                    const auto game = entry.toObject();
                    games->addItem(game.value(QStringLiteral("title")).toString(),
                                   game.value(QStringLiteral("sgdbId")).toInteger());
                }
                page_count = pages;
                status->setText(tr("Page %1 of %2. Select the exact game and press Confirm game.")
                                    .arg(page)
                                    .arg(page_count));
            } else {
                status->setText(
                    tr("The catalog returned an invalid response. No game was confirmed."));
            }
        }
    }
    if (submission) {
        public_consent->setChecked(false);
        safe_files_consent->setChecked(false);
    }
    Refresh();
}

} // namespace CompatibilityReporting
