// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QByteArray>
#include <QDialog>
#include <QList>
#include <QMap>
#include <QPointer>
#include <QString>
#include <QUrl>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QNetworkAccessManager;
class QNetworkReply;
class QPlainTextEdit;
class QPushButton;
class QTabWidget;

namespace CompatibilityReporting {

struct Addon {
    QString name;
    QString version;
    bool enabled;
};

QString SummarizeEnabledAddons(const QList<Addon>& addons);

struct Session {
    quint64 program_id{};
    QString title;
    QString build;
    QString game_version;
    QString cpu;
    QString gpu;
    QString driver;
    QString os;
    QString ram;
    QString graphics_api;
    QString settings;
    QString mods;
    QString updates;
};

class Dialog final : public QDialog {
    Q_OBJECT

public:
    explicit Dialog(const Session& session, QWidget* parent = nullptr);
    ~Dialog() override;
    void reject() override;

private:
    friend class DialogTest;
    Dialog(const Session& session, QNetworkAccessManager* network, QWidget* parent);
    enum class State { Editing, Searching, Submitting, Finished, Closed };
    void Search();
    void ConfirmGame();
    void Submit();
    void Clear();
    void Edited();
    void Refresh();
    void Abort();
    void Request(bool submission, const QByteArray& payload = {});
    void Finished(bool submission, int status, const QByteArray& body,
                  const QByteArray& retry_after, bool network_error);
    QByteArray Payload() const;
    QString ValidationError() const;

    QNetworkAccessManager* network;
    QPointer<QNetworkReply> reply;
    State state{State::Editing};
    qint64 confirmed_game{};
    QString confirmed_title;
    QByteArray submitted_payload;
    QByteArray submission_key;
    qint64 retry_at{};
    int page{1};
    int page_count{1};
    QUrl receipt_url;
    QMap<QString, QLineEdit*> fields;
    QMap<QString, QPlainTextEdit*> paragraphs;
    QLineEdit* query;
    QComboBox* games;
    QComboBox* outcome;
    QComboBox* graphics_api;
    QCheckBox* public_consent;
    QCheckBox* safe_files_consent;
    QPushButton* search;
    QPushButton* confirm;
    QPushButton* previous;
    QPushButton* next;
    QPushButton* submit;
    QPushButton* clear;
    QPushButton* open_receipt;
    QLabel* status;
    QLabel* match;
    QTabWidget* tabs;
    QPlainTextEdit* preview;
};

} // namespace CompatibilityReporting
