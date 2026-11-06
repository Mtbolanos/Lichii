// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/compatibility_report.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkCookie>
#include <QNetworkCookieJar>
#include <QNetworkReply>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QTabWidget>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTest>
#include <QTextCursor>
#include <QTranslator>
#include <QUrlQuery>

namespace CompatibilityReporting {

class FakeReply final : public QNetworkReply {
public:
    FakeReply(const QNetworkRequest& request, QObject* parent) : QNetworkReply(parent) {
        setRequest(request);
        setUrl(request.url());
        open(QIODevice::ReadOnly | QIODevice::Unbuffered);
    }

    void abort() override {
        aborted = true;
        setError(OperationCanceledError, QStringLiteral("Cancelled"));
        emit finished();
    }

    void Complete(int status, QByteArray bytes, QByteArray retry_after = {}) {
        data = std::move(bytes);
        setAttribute(QNetworkRequest::HttpStatusCodeAttribute, status);
        setRawHeader("Retry-After", retry_after);
        emit readyRead();
        setFinished(true);
        emit finished();
    }

    qint64 bytesAvailable() const override {
        return data.size() - offset + QNetworkReply::bytesAvailable();
    }

    bool aborted{};

protected:
    qint64 readData(char* output, qint64 maximum) override {
        const auto count = std::min(maximum, qint64(data.size() - offset));
        if (count <= 0) {
            return -1;
        }
        std::memcpy(output, data.constData() + offset, size_t(count));
        offset += count;
        return count;
    }

private:
    QByteArray data;
    qint64 offset{};
};

class FakeNetwork final : public QNetworkAccessManager {
public:
    struct Request {
        Operation operation;
        QNetworkRequest request;
        QByteArray body;
        QPointer<FakeReply> reply;
    };
    QList<Request> requests;

protected:
    QNetworkReply* createRequest(Operation operation, const QNetworkRequest& request,
                                 QIODevice* data) override {
        auto* reply = new FakeReply(request, this);
        requests.push_back({operation, request, data ? data->readAll() : QByteArray{}, reply});
        return reply;
    }
};

class DialogTest final : public QObject {
    Q_OBJECT

public:
    static Session Seed();
    static QByteArray Catalog(int page = 1, int pages = 1);
    static void Prepare(Dialog& dialog, FakeNetwork& network);
    static int Preview(QApplication& app, const QString& screenshot = {});

private slots:
    void noImplicitTraffic();
    void explicitCatalogAndPaging();
    void consentAndDoubleSubmit();
    void retryIdentityAndEdits();
    void rateLimitAndCancellation();
    void invalidCatalog_data();
    void invalidCatalog();
    void unsafeReceiptAndRedirect();
    void localValidationAndClear();
    void isolatedHttpTransport();
    void serverRejections_data();
    void serverRejections();
    void reconfirmationKeepsRetryIdentity();
    void cooldownPreservesPreviewSelection();
    void dialogTranslationContext();
    void machinePrefillsRemainEditable();
    void enabledAddonPrefills();
};

Session DialogTest::Seed() {
    return {.program_id = 0x0400000000000000ULL,
            .title = QStringLiteral("Synthetic Switch 2 game"),
            .build = QStringLiteral("Lichii prototype (synthetic)"),
            .game_version = QStringLiteral("1.0.0"),
            .cpu = QStringLiteral("Example CPU"),
            .gpu = {},
            .driver = {},
            .os = QStringLiteral("Example OS"),
            .ram = QStringLiteral("16 GiB"),
            .graphics_api = QStringLiteral("Vulkan"),
            .settings = QStringLiteral("Launch settings: handheld mode; resolution scale 1x."),
            .mods = {},
            .updates = {}};
}

QByteArray DialogTest::Catalog(int page, int pages) {
    return QJsonDocument(
               QJsonObject{
                   {QStringLiteral("games"),
                    QJsonArray{QJsonObject{
                        {QStringLiteral("sgdbId"), 12345},
                        {QStringLiteral("slug"), QStringLiteral("synthetic-game")},
                        {QStringLiteral("title"), QStringLiteral("Synthetic Switch 2 game")}}}},
                   {QStringLiteral("page"), page},
                   {QStringLiteral("pageCount"), pages},
                   {QStringLiteral("total"), 1}})
        .toJson();
}

void DialogTest::Prepare(Dialog& dialog, FakeNetwork& network) {
    dialog.search->click();
    network.requests.last().reply->Complete(200, Catalog());
    dialog.confirm->click();
    dialog.fields[QStringLiteral("summary")]->setText(
        QStringLiteral("Synthetic test, never sent to the Internet"));
    dialog.paragraphs[QStringLiteral("progress")]->setPlainText(
        QStringLiteral("Played the opening area in a synthetic session."));
    dialog.paragraphs[QStringLiteral("details")]->setPlainText(
        QStringLiteral("Example report for local UI verification only."));
    dialog.outcome->setCurrentIndex(2);
}

int DialogTest::Preview(QApplication& app, const QString& screenshot) {
    FakeNetwork network;
    auto session = Seed();
    session.gpu = QStringLiteral("Example active GPU");
    session.driver = QStringLiteral("Example driver 1.2.3");
    Dialog dialog(session, &network, nullptr);
    Prepare(dialog, network);
    dialog.status->setText(
        QStringLiteral("LOCAL MOCK ONLY. No report or catalog request reaches the Internet."));
    dialog.show();
    if (!screenshot.isEmpty()) {
        dialog.tabs->setCurrentIndex(1);
        QTest::qWait(100);
        return dialog.grab().save(screenshot) ? 0 : 1;
    }
    return app.exec();
}

void DialogTest::noImplicitTraffic() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    QCOMPARE(dialog.query->text(), Seed().title);
    QVERIFY(network.requests.empty());
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    QVERIFY(!dialog.submit->isEnabled());
    dialog.query->setText(QStringLiteral("A different title"));
    dialog.fields[QStringLiteral("name")]->setText(QStringLiteral("Player"));
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    dialog.clear->click();
    dialog.reject();
    QCOMPARE(network.requests.size(), 0);
}

void DialogTest::explicitCatalogAndPaging() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    dialog.search->click();
    QCOMPARE(network.requests.size(), 1);
    const auto& request = network.requests.first().request;
    QCOMPARE(request.url().path(), QStringLiteral("/api/games"));
    QCOMPARE(QUrlQuery(request.url()).queryItemValue(QStringLiteral("sort")),
             QStringLiteral("title"));
    QVERIFY(!request.hasRawHeader("X-Lichii-Client"));
    network.requests.last().reply->Complete(200, Catalog(1, 2));
    QCOMPARE(dialog.confirmed_game, 0);
    dialog.confirm->click();
    QCOMPARE(dialog.confirmed_game, 12345);
    QCOMPARE(network.requests.size(), 1);
    dialog.next->click();
    QCOMPARE(network.requests.size(), 2);
    QCOMPARE(
        QUrlQuery(network.requests.last().request.url()).queryItemValue(QStringLiteral("page")),
        QStringLiteral("2"));
    QCOMPARE(dialog.confirmed_game, 0);
    network.requests.last().reply->Complete(200, Catalog(2, 2));
    QVERIFY(!dialog.next->isEnabled());
    dialog.previous->click();
    QCOMPARE(network.requests.size(), 3);
    auto pending = network.requests.last().reply;
    dialog.query->setText(QStringLiteral("Changed title"));
    QVERIFY(pending->aborted);
    pending->Complete(200, Catalog());
    QCOMPARE(dialog.games->count(), 0);
    QCOMPARE(dialog.confirmed_game, 0);
}

void DialogTest::consentAndDoubleSubmit() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    QVERIFY(dialog.ValidationError().isEmpty());
    dialog.Submit();
    QCOMPARE(network.requests.size(), 1);
    dialog.public_consent->setChecked(true);
    dialog.Submit();
    QCOMPARE(network.requests.size(), 1);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    dialog.Submit();
    QCOMPARE(network.requests.size(), 2);
    QVERIFY(!dialog.submit->isEnabled());
    const auto& sent = network.requests.last();
    QCOMPARE(sent.operation, QNetworkAccessManager::PostOperation);
    QCOMPARE(sent.request.url().toString(),
             QStringLiteral("https://lichii.zortos.me/api/v1/desktop/reports"));
    QCOMPARE(sent.request.rawHeader("X-Lichii-Client"), QByteArray("desktop/1"));
    QVERIFY(!QUuid(QString::fromUtf8(sent.request.rawHeader("Idempotency-Key"))).isNull());
    for (const auto& header : {"Cookie", "Authorization", "Origin", "Sec-Fetch-Site"}) {
        QVERIFY(!sent.request.hasRawHeader(header));
    }
    QCOMPARE(sent.request.attribute(QNetworkRequest::CookieLoadControlAttribute).toInt(),
             int(QNetworkRequest::Manual));
    QCOMPARE(sent.request.attribute(QNetworkRequest::CookieSaveControlAttribute).toInt(),
             int(QNetworkRequest::Manual));
    QCOMPARE(sent.request.attribute(QNetworkRequest::AuthenticationReuseAttribute).toInt(),
             int(QNetworkRequest::Manual));
    QCOMPARE(sent.request.attribute(QNetworkRequest::RedirectPolicyAttribute).toInt(),
             int(QNetworkRequest::ManualRedirectPolicy));
    const auto object = QJsonDocument::fromJson(sent.body).object();
    const QStringList expected_fields{
        QStringLiteral("build"),         QStringLiteral("cpu"),
        QStringLiteral("details"),       QStringLiteral("driver"),
        QStringLiteral("gameId"),        QStringLiteral("gameVersion"),
        QStringLiteral("gpu"),           QStringLiteral("graphicsApi"),
        QStringLiteral("handle"),        QStringLiteral("mods"),
        QStringLiteral("name"),          QStringLiteral("os"),
        QStringLiteral("outcome"),       QStringLiteral("progress"),
        QStringLiteral("publicConsent"), QStringLiteral("ram"),
        QStringLiteral("reproduction"),  QStringLiteral("safeFilesConsent"),
        QStringLiteral("settings"),      QStringLiteral("summary"),
        QStringLiteral("updates")};
    QCOMPARE(object.keys(), expected_fields);
    QCOMPARE(object.value(QStringLiteral("gameId")).toInt(), 12345);
    QVERIFY(object.value(QStringLiteral("publicConsent")).toBool());
    QVERIFY(object.value(QStringLiteral("safeFilesConsent")).toBool());
    for (const auto& key : {"log", "privateLogConsent", "website", "programId", "title", "path"}) {
        QVERIFY(!object.contains(QString::fromLatin1(key)));
    }
    QCOMPARE(object.value(QStringLiteral("gpu")).toString(), QStringLiteral("Unknown"));
    network.requests.last().reply->Complete(
        201,
        R"({"id":"12345678-1234-4234-8234-123456789abc","url":"https://lichii.zortos.me/reports/12345678-1234-4234-8234-123456789abc"})");
    QCOMPARE(dialog.state, Dialog::State::Finished);
    dialog.Submit();
    QCOMPARE(network.requests.size(), 2);
}

void DialogTest::retryIdentityAndEdits() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    const auto original = network.requests.last();
    original.reply->Complete(503, "{}");
    QCOMPARE(network.requests.size(), 2);
    QVERIFY(!dialog.public_consent->isChecked());
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    QCOMPARE(network.requests.size(), 3);
    QCOMPARE(network.requests.last().body, original.body);
    QCOMPARE(network.requests.last().request.rawHeader("Idempotency-Key"),
             original.request.rawHeader("Idempotency-Key"));
    network.requests.last().reply->Complete(400, R"({"fields":{"summary":["Too long"]}})");
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.fields[QStringLiteral("summary")]->setText(QStringLiteral("Edited synthetic draft"));
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    QVERIFY(network.requests.last().request.rawHeader("Idempotency-Key") !=
            original.request.rawHeader("Idempotency-Key"));
}

void DialogTest::rateLimitAndCancellation() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    network.requests.last().reply->Complete(429, "{}", "1");
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    QVERIFY(!dialog.submit->isEnabled());
    dialog.Submit();
    QCOMPARE(network.requests.size(), 2);
    QTest::qWait(1100);
    dialog.Refresh();
    QVERIFY(dialog.submit->isEnabled());
    dialog.Submit();
    auto pending = network.requests.last().reply;
    dialog.reject();
    QVERIFY(pending->aborted);
    pending->Complete(201, "{}");
    QCOMPARE(dialog.state, Dialog::State::Closed);
    QCOMPARE(network.requests.size(), 3);
}

void DialogTest::invalidCatalog_data() {
    QTest::addColumn<QByteArray>("body");
    QTest::newRow("not-json") << QByteArray("invalid");
    QTest::newRow("wrong-id") << QByteArray(
        R"({"games":[{"sgdbId":"123","title":"Title","slug":"title"}],"page":1,"pageCount":1})");
    QTest::newRow("fractional-id") << QByteArray(
        R"({"games":[{"sgdbId":1.5,"title":"Title","slug":"title"}],"page":1,"pageCount":1})");
    QTest::newRow("missing-games") << QByteArray(R"({"page":1,"pageCount":1})");
    QTest::newRow("oversized") << QByteArray(256 * 1024 + 1, 'x');
}

void DialogTest::invalidCatalog() {
    QFETCH(QByteArray, body);
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    dialog.search->click();
    network.requests.last().reply->Complete(200, body);
    QCOMPARE(dialog.games->count(), 0);
    QCOMPARE(dialog.confirmed_game, 0);
    QVERIFY(!dialog.submit->isEnabled());
}

void DialogTest::unsafeReceiptAndRedirect() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    for (const auto http_status : {302, 201}) {
        dialog.public_consent->setChecked(true);
        dialog.safe_files_consent->setChecked(true);
        dialog.Submit();
        network.requests.last().reply->Complete(
            http_status,
            R"({"id":"12345678-1234-4234-8234-123456789abc","url":"https://evil.example/reports/12345678-1234-4234-8234-123456789abc"})");
        QVERIFY(dialog.receipt_url.isEmpty());
        QCOMPARE(dialog.state, Dialog::State::Editing);
    }
    QCOMPARE(network.requests.size(), 3);
}

void DialogTest::localValidationAndClear() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.fields[QStringLiteral("handle")]->setText(QStringLiteral("not valid!"));
    QVERIFY(!dialog.ValidationError().isEmpty());
    dialog.fields[QStringLiteral("handle")]->clear();
    dialog.paragraphs[QStringLiteral("details")]->setPlainText(QString(6001, QLatin1Char('x')));
    QVERIFY(!dialog.ValidationError().isEmpty());
    dialog.paragraphs[QStringLiteral("details")]->setPlainText(QStringLiteral("Text") + QChar(1));
    QVERIFY(!dialog.ValidationError().isEmpty());
    dialog.clear->click();
    QCOMPARE(dialog.confirmed_game, 0);
    QVERIFY(dialog.fields[QStringLiteral("cpu")]->text().isEmpty());
    QVERIFY(dialog.paragraphs[QStringLiteral("details")]->toPlainText().isEmpty());
    QCOMPARE(network.requests.size(), 1);
}

void DialogTest::isolatedHttpTransport() {
    QTcpServer server;
    QVERIFY(server.listen(QHostAddress::LocalHost));
    class LocalNetwork final : public QNetworkAccessManager {
    public:
        quint16 port{};
        int count{};

    protected:
        QNetworkReply* createRequest(Operation operation, const QNetworkRequest& request,
                                     QIODevice* body) override {
            auto local = request;
            auto url = local.url();
            if (url.host() != QStringLiteral("lichii.zortos.me")) {
                qFatal("Unexpected request destination in local-only test");
            }
            url.setScheme(QStringLiteral("http"));
            url.setHost(QStringLiteral("127.0.0.1"));
            url.setPort(port);
            local.setUrl(url);
            ++count;
            return QNetworkAccessManager::createRequest(operation, local, body);
        }
    } network;
    network.port = server.serverPort();
    network.cookieJar()->setCookiesFromUrl(
        {QNetworkCookie("test-session", "must-not-be-sent")},
        QUrl(QStringLiteral("http://127.0.0.1:%1/").arg(network.port)));
    QList<QByteArray> wire_requests;
    connect(&server, &QTcpServer::newConnection, &server, [&] {
        auto* socket = server.nextPendingConnection();
        auto bytes = std::make_shared<QByteArray>();
        connect(socket, &QTcpSocket::readyRead, &server, [&, socket, bytes] {
            bytes->append(socket->readAll());
            const auto split = bytes->indexOf("\r\n\r\n");
            if (split < 0) {
                return;
            }
            qsizetype content_length = 0;
            for (const auto& line : bytes->left(split).split('\n')) {
                if (line.toLower().startsWith("content-length:")) {
                    content_length = line.mid(line.indexOf(':') + 1).trimmed().toLongLong();
                }
            }
            if (bytes->size() < split + 4 + content_length) {
                return;
            }
            wire_requests.append(*bytes);
            const QByteArray body =
                bytes->startsWith("GET ")
                    ? Catalog()
                    : QByteArray(
                          R"({"id":"12345678-1234-4234-8234-123456789abc","url":"https://lichii.zortos.me/reports/12345678-1234-4234-8234-123456789abc"})");
            socket->write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: "
                          "close\r\nContent-Length: " +
                          QByteArray::number(body.size()) + "\r\n\r\n" + body);
            socket->disconnectFromHost();
        });
    });
    Dialog dialog(Seed(), &network, nullptr);
    QTest::qWait(20);
    QCOMPARE(network.count, 0);
    const auto query_text = QStringLiteral("Café + %2B & #?");
    dialog.query->setText(QStringLiteral("  %1  ").arg(query_text));
    dialog.search->click();
    QTRY_COMPARE(dialog.games->count(), 1);
    dialog.confirm->click();
    dialog.fields[QStringLiteral("summary")]->setText(QStringLiteral("Loopback fixture only"));
    dialog.paragraphs[QStringLiteral("progress")]->setPlainText(QStringLiteral("Opening area"));
    dialog.paragraphs[QStringLiteral("details")]->setPlainText(
        QStringLiteral("Synthetic local fixture"));
    dialog.outcome->setCurrentIndex(2);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    QCOMPARE(network.count, 1);
    dialog.submit->click();
    dialog.Submit();
    QTRY_COMPARE(dialog.state, Dialog::State::Finished);
    QCOMPARE(network.count, 2);
    QCOMPARE(wire_requests.size(), 2);
    const auto target = wire_requests.first().split(' ').value(1);
    QByteArray encoded_query;
    for (const auto& item : target.mid(target.indexOf('?') + 1).split('&')) {
        if (item.startsWith("q=")) {
            encoded_query = item.mid(2);
        }
    }
    QCOMPARE(QUrl::fromPercentEncoding(encoded_query.replace('+', ' ')), query_text);
    for (const auto& request : wire_requests) {
        const auto headers = request.left(request.indexOf("\r\n\r\n")).toLower();
        QVERIFY(!headers.contains("\r\ncookie:"));
        QVERIFY(!headers.contains("\r\nauthorization:"));
        QVERIFY(!headers.contains("\r\norigin:"));
        QVERIFY(!headers.contains("\r\nsec-fetch-"));
    }
    const auto request = wire_requests.last();
    QMap<QByteArray, QByteArray> headers;
    for (const auto& line : request.left(request.indexOf("\r\n\r\n")).split('\n')) {
        const auto colon = line.indexOf(':');
        if (colon > 0) {
            headers.insert(line.left(colon).trimmed().toLower(), line.mid(colon + 1).trimmed());
        }
    }
    QCOMPARE(headers.value("x-lichii-client"), QByteArray("desktop/1"));
    const auto body = request.mid(request.indexOf("\r\n\r\n") + 4);
    QCOMPARE(body, dialog.submitted_payload);
    QVERIFY(!body.contains("must-not-be-sent"));
}

void DialogTest::serverRejections_data() {
    QTest::addColumn<int>("code");
    for (const int code : {400, 403, 409, 413, 415, 500, 503}) {
        QTest::newRow(QByteArray::number(code).constData()) << code;
    }
}

void DialogTest::serverRejections() {
    QFETCH(int, code);
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    network.requests.last().reply->Complete(
        code,
        R"({"error":"<script>untrusted</script>","fields":{"summary":["Untrusted server text"]}})");
    QCOMPARE(dialog.state, Dialog::State::Editing);
    QCOMPARE(network.requests.size(), 2);
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    QVERIFY(!dialog.status->text().contains(QStringLiteral("untrusted"), Qt::CaseInsensitive));
    QVERIFY(!dialog.fields[QStringLiteral("summary")]->text().isEmpty());
    if (code == 400) {
        QVERIFY(dialog.status->text().contains(QStringLiteral("summary")));
    }
}

void DialogTest::reconfirmationKeepsRetryIdentity() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    const auto original = network.requests.last();
    original.reply->Complete(503, "{}");
    dialog.confirm->click();
    QCOMPARE(network.requests.size(), 2);
    QCOMPARE(dialog.submission_key, original.request.rawHeader("Idempotency-Key"));
    QCOMPARE(dialog.submitted_payload, original.body);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    QCOMPARE(network.requests.last().request.rawHeader("Idempotency-Key"),
             original.request.rawHeader("Idempotency-Key"));
    QCOMPARE(network.requests.last().body, original.body);
    network.requests.last().reply->Complete(503, "{}");
    dialog.games->addItem(QStringLiteral("Different catalog game"), 98765);
    dialog.games->setCurrentIndex(1);
    dialog.confirm->click();
    QVERIFY(dialog.submission_key.isEmpty());
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    QVERIFY(network.requests.last().request.rawHeader("Idempotency-Key") !=
            original.request.rawHeader("Idempotency-Key"));
}

void DialogTest::cooldownPreservesPreviewSelection() {
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    Prepare(dialog, network);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.Submit();
    network.requests.last().reply->Complete(429, "{}", "60");
    dialog.tabs->setCurrentWidget(dialog.preview);
    dialog.show();
    QTest::qWait(20);
    auto cursor = dialog.preview->textCursor();
    cursor.setPosition(20);
    cursor.setPosition(80, QTextCursor::KeepAnchor);
    dialog.preview->setTextCursor(cursor);
    auto* scroll = dialog.preview->verticalScrollBar();
    scroll->setValue(scroll->maximum());
    const auto position = scroll->value();
    const auto selected = cursor.selectedText();
    QTest::qWait(1200);
    QCOMPARE(dialog.preview->textCursor().selectedText(), selected);
    QCOMPARE(scroll->value(), position);
    QVERIFY(!dialog.submit->isEnabled());
    QCOMPARE(network.requests.size(), 2);
}

void DialogTest::dialogTranslationContext() {
    class Translator final : public QTranslator {
    public:
        QString translate(const char* context, const char* source, const char*,
                          int) const override {
            if (QByteArray(context) == "CompatibilityReporting::Dialog" &&
                QByteArray(source) == "Submit compatibility report") {
                return QStringLiteral("Translated report title");
            }
            return {};
        }
        bool isEmpty() const override {
            return false;
        }
    } translator;
    QCoreApplication::installTranslator(&translator);
    FakeNetwork network;
    Dialog dialog(Seed(), &network, nullptr);
    const auto title = dialog.windowTitle();
    QCoreApplication::removeTranslator(&translator);
    QCOMPARE(title, QStringLiteral("Translated report title"));
}

void DialogTest::machinePrefillsRemainEditable() {
    FakeNetwork network;
    auto session = Seed();
    session.gpu = QStringLiteral(" Active rendering GPU ");
    session.driver = QStringLiteral(" Driver 123.45 ");
    Dialog dialog(session, &network, nullptr);
    QCOMPARE(network.requests.size(), 0);
    QCOMPARE(dialog.fields[QStringLiteral("gpu")]->text(), QStringLiteral("Active rendering GPU"));
    QCOMPARE(dialog.fields[QStringLiteral("driver")]->text(), QStringLiteral("Driver 123.45"));
    QCOMPARE(dialog.graphics_api->currentText(), QStringLiteral("Vulkan"));
    QCOMPARE(dialog.fields[QStringLiteral("cpu")]->text(), session.cpu);
    QCOMPARE(dialog.fields[QStringLiteral("ram")]->text(), session.ram);
    QCOMPARE(dialog.fields[QStringLiteral("os")]->text(), session.os);
    session.gpu.clear();
    session.driver.clear();
    QCOMPARE(dialog.fields[QStringLiteral("gpu")]->text(), QStringLiteral("Active rendering GPU"));
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.fields[QStringLiteral("driver")]->setText(QStringLiteral("Corrected driver"));
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    const auto payload = QJsonDocument::fromJson(dialog.Payload()).object();
    QCOMPARE(payload.value(QStringLiteral("driver")).toString(),
             QStringLiteral("Corrected driver"));
    QCOMPARE(payload.value(QStringLiteral("gpu")).toString(),
             QStringLiteral("Active rendering GPU"));
    QCOMPARE(network.requests.size(), 0);
}

void DialogTest::enabledAddonPrefills() {
    auto session = Seed();
    session.mods = SummarizeEnabledAddons({
        {QStringLiteral("Dynamic FPS"), QStringLiteral("IPS"), true},
        {QStringLiteral("Max Resolution"), QStringLiteral("IPS"), false},
        {QStringLiteral("Dynamic FPS"), QStringLiteral("IPS"), true},
    });
    session.updates = SummarizeEnabledAddons({
        {QStringLiteral("Update"), QStringLiteral("2.0.0"), true},
        {QStringLiteral("Update"), QStringLiteral("1.0.0"), false},
    });
    QCOMPARE(session.mods, QStringLiteral("Enabled at launch: Dynamic FPS (IPS)"));
    QCOMPARE(session.updates, QStringLiteral("Enabled at launch: Update (2.0.0)"));
    QCOMPARE(SummarizeEnabledAddons({}), QStringLiteral("None listed as enabled at launch."));
    const auto long_summary = SummarizeEnabledAddons({{QString(700, QLatin1Char('x')), {}, true}});
    QVERIFY(long_summary.size() <= 500);
    QVERIFY(long_summary.endsWith(QStringLiteral("(list truncated; review)")));
    FakeNetwork network;
    Dialog dialog(session, &network, nullptr);
    QCOMPARE(dialog.paragraphs[QStringLiteral("mods")]->toPlainText(), session.mods);
    QCOMPARE(dialog.paragraphs[QStringLiteral("updates")]->toPlainText(), session.updates);
    const auto object = QJsonDocument::fromJson(dialog.Payload()).object();
    QCOMPARE(object.value(QStringLiteral("mods")).toString(), session.mods);
    QCOMPARE(object.value(QStringLiteral("updates")).toString(), session.updates);
    dialog.public_consent->setChecked(true);
    dialog.safe_files_consent->setChecked(true);
    dialog.paragraphs[QStringLiteral("mods")]->setPlainText(QStringLiteral("Corrected mod list"));
    QVERIFY(!dialog.public_consent->isChecked());
    QVERIFY(!dialog.safe_files_consent->isChecked());
    QCOMPARE(network.requests.size(), 0);
}

} // namespace CompatibilityReporting

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    if (app.arguments().contains(QStringLiteral("--preview"))) {
        return CompatibilityReporting::DialogTest::Preview(app);
    }
    const int screenshot = app.arguments().indexOf(QStringLiteral("--screenshot"));
    if (screenshot >= 0 && screenshot + 1 < app.arguments().size()) {
        return CompatibilityReporting::DialogTest::Preview(app, app.arguments()[screenshot + 1]);
    }
    CompatibilityReporting::DialogTest test;
    return QTest::qExec(&test, argc, argv);
}

#include "compatibility_report.moc"
