// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "common/httplib.h"
#include "common/net/net.h"
#include "frontend_common/update_checker.h"

namespace {

Common::Net::Release MakeRelease(std::string tag, std::string filename) {
    const nlohmann::json json{
        {"tag_name", tag},
        {"name", "Lichii nx2 " + tag},
        {"prerelease", true},
        {"assets",
         {{{"browser_download_url",
            "https://github.com/zortos293/Lichii/releases/download/" + tag + "/" + filename}}}},
    };
    return Common::Net::Release::FromJson(json, "https://api.github.com", "zortos293/Lichii")
        .value();
}

} // namespace

TEST_CASE("Lichii updates compare build numbers, not tag strings", "[updater]") {
    const auto releases = std::array{
        MakeRelease("build-9", "lichii-windows-x86_64.zip"),
        MakeRelease("build-100", "lichii-windows-x86_64.zip"),
        MakeRelease("build-99", "lichii-windows-x86_64.zip"),
    };
    const auto update = UpdateChecker::SelectUpdate(releases, "build-70", "windows-x86_64");
    REQUIRE(update);
    CHECK(update->tag == "build-100");
    CHECK_FALSE(UpdateChecker::SelectUpdate(releases, "build-100", "windows-x86_64"));
    CHECK_FALSE(UpdateChecker::SelectUpdate(releases, "build-101", "windows-x86_64"));
    CHECK_FALSE(UpdateChecker::SelectUpdate({}, "build-70", "windows-x86_64"));
}

TEST_CASE("Lichii ignores dev builds and malformed release tags", "[updater]") {
    const std::string tag =
        GENERATE("", "nx2", "v0.2.0", "abc123-nx2", "nx2-build-71", "build-", "build-0", "build--1",
                 "build-+71", "build-71junk", "build- 71", "build-18446744073709551616");
    const auto releases = std::array{MakeRelease("build-71", "lichii-windows-x86_64.zip")};
    CHECK_FALSE(UpdateChecker::SelectUpdate(releases, tag, "windows-x86_64"));
    auto invalid = releases;
    invalid[0].tag = tag;
    invalid[0].assets = {"https://github.com/zortos293/Lichii/releases/download/" + tag +
                         "/lichii-windows-x86_64.zip"};
    CHECK_FALSE(UpdateChecker::SelectUpdate(invalid, "build-70", "windows-x86_64"));
}

TEST_CASE("Lichii selects only the matching platform and PGO variant", "[updater]") {
    const std::string platform =
        GENERATE("windows-x86_64", "windows-x86_64-pgo", "linux-x86_64", "linux-x86_64-pgo",
                 "linux-aarch64", "linux-aarch64-pgo", "macos-arm64");
    auto release = MakeRelease("build-71", "unrelated.zip");
    for (const auto name :
         {"windows-x86_64.zip", "windows-x86_64-pgo.zip", "linux-x86_64.tar.gz",
          "linux-x86_64-pgo.tar.gz", "linux-aarch64.tar.gz", "linux-aarch64-pgo.tar.gz",
          "macos-arm64.zip", "linux-x86_64.AppImage", "linux-x86_64-pgo.AppImage",
          "linux-aarch64.AppImage", "linux-aarch64-pgo.AppImage"}) {
        release.assets.push_back(
            "https://github.com/zortos293/Lichii/releases/download/build-71/lichii-" +
            std::string{name});
    }
    const auto selected = release.GetPlatformAssets(platform);
    REQUIRE(selected.size() == 1);
    CHECK(selected[0].filename ==
          "lichii-" + platform + (platform.starts_with("linux-") ? ".AppImage" : ".zip"));
    CHECK(selected[0].url == "https://github.com");
    CHECK(selected[0].path ==
          "/zortos293/Lichii/releases/download/build-71/" + selected[0].filename);
    CHECK_FALSE(release.GetPlatformAssets("android-arm64").size());
    CHECK_FALSE(release.GetPlatformAssets("").size());
}

TEST_CASE("Lichii skips releases without its asset and falls back to Linux archives", "[updater]") {
    const auto releases = std::array{
        MakeRelease("build-73", "lichii-linux-x86_64.AppImage"),
        MakeRelease("build-71", "lichii-linux-x86_64-pgo.tar.gz"),
        MakeRelease("build-72", "lichii-linux-aarch64-pgo.AppImage"),
    };
    const auto update = UpdateChecker::SelectUpdate(releases, "build-70", "linux-x86_64");
    REQUIRE(update);
    CHECK(update->tag == "build-73");
    CHECK(update->GetPlatformAssets("linux-x86_64")[0].filename == "lichii-linux-x86_64.AppImage");
    CHECK_FALSE(UpdateChecker::SelectUpdate(releases, "build-70", "windows-x86_64"));

    const auto archive = UpdateChecker::SelectUpdate(releases, "build-70", "linux-x86_64-pgo",
                                                     Common::Net::LinuxPackage::Archive);
    REQUIRE(archive);
    CHECK(archive->tag == "build-71");
    CHECK(archive->GetPlatformAssets("linux-x86_64-pgo", Common::Net::LinuxPackage::Archive)[0]
              .filename == "lichii-linux-x86_64-pgo.tar.gz");
}

TEST_CASE("PGO builds fall back to the standard asset, standard builds never get PGO", "[updater]") {
    auto release = MakeRelease("build-71", "lichii-windows-x86_64.zip");
    const auto pgo = release.GetPlatformAssets("windows-x86_64-pgo");
    REQUIRE(pgo.size() == 1);
    CHECK(pgo[0].name == "Standard");
    CHECK(pgo[0].filename == "lichii-windows-x86_64.zip");

    release.assets.push_back(
        "https://github.com/zortos293/Lichii/releases/download/build-71/lichii-windows-x86_64-pgo.zip");
    const auto preferred = release.GetPlatformAssets("windows-x86_64-pgo");
    REQUIRE(preferred.size() == 1);
    CHECK(preferred[0].name == "PGO");
    CHECK(preferred[0].filename == "lichii-windows-x86_64-pgo.zip");
    CHECK(release.GetPlatformAssets("windows-x86_64")[0].filename == "lichii-windows-x86_64.zip");

    const auto pgo_only = MakeRelease("build-72", "lichii-windows-x86_64-pgo.zip");
    CHECK(pgo_only.GetPlatformAssets("windows-x86_64").empty());
    CHECK_FALSE(UpdateChecker::SelectUpdate(std::array{pgo_only}, "build-70", "windows-x86_64"));
    CHECK(UpdateChecker::SelectUpdate(std::array{pgo_only}, "build-70", "windows-x86_64-pgo"));
    CHECK(release.GetPlatformAssets("macos-arm64-pgo").empty());
}

TEST_CASE("Linux installs only get the package they were installed from", "[updater]") {
    auto release = MakeRelease("build-71", "lichii-linux-x86_64.AppImage");
    release.assets.push_back(
        "https://github.com/zortos293/Lichii/releases/download/build-71/lichii-linux-x86_64.tar.gz");
    CHECK(release.GetPlatformAssets("linux-x86_64", Common::Net::LinuxPackage::AppImage)[0]
              .filename == "lichii-linux-x86_64.AppImage");
    CHECK(release.GetPlatformAssets("linux-x86_64", Common::Net::LinuxPackage::Archive)[0]
              .filename == "lichii-linux-x86_64.tar.gz");

    const auto appimage_only = MakeRelease("build-72", "lichii-linux-x86_64.AppImage");
    CHECK(appimage_only.GetPlatformAssets("linux-x86_64", Common::Net::LinuxPackage::Archive)
              .empty());
}

TEST_CASE("Asset digest and size come from the release JSON", "[updater]") {
    const std::string sha(64, 'A');
    const nlohmann::json json{
        {"tag_name", "build-71"},
        {"assets",
         {{{"browser_download_url",
            "https://github.com/zortos293/Lichii/releases/download/build-71/"
            "lichii-windows-x86_64.zip"},
           {"size", u64{1234}},
           {"digest", "sha256:" + sha}},
          {{"browser_download_url",
            "https://github.com/zortos293/Lichii/releases/download/build-71/"
            "lichii-linux-x86_64.AppImage"},
           {"size", u64{5}},
           {"digest", "sha256:zz"}},
          {{"browser_download_url",
            "https://github.com/zortos293/Lichii/releases/download/build-71/"
            "lichii-linux-x86_64.tar.gz"},
           {"digest", "md5:abc"}}}},
    };
    const auto release =
        Common::Net::Release::FromJson(json, "https://api.github.com", "zortos293/Lichii").value();
    const auto windows = release.GetPlatformAssets("windows-x86_64");
    REQUIRE(windows.size() == 1);
    CHECK(windows[0].sha256 == std::string(64, 'a'));
    CHECK(windows[0].size == 1234);
    const auto appimage = release.GetPlatformAssets("linux-x86_64");
    REQUIRE(appimage.size() == 1);
    CHECK(appimage[0].sha256.empty());
    CHECK(appimage[0].size == 5);
    const auto archive =
        release.GetPlatformAssets("linux-x86_64", Common::Net::LinuxPackage::Archive);
    REQUIRE(archive.size() == 1);
    CHECK(archive[0].sha256.empty());
    CHECK(archive[0].size == 0);
}

TEST_CASE("Failed release requests are classified by status", "[updater]") {
    CHECK(UpdateChecker::ClassifyFailure(403) == UpdateChecker::Failure::RateLimited);
    CHECK(UpdateChecker::ClassifyFailure(429) == UpdateChecker::Failure::RateLimited);
    CHECK(UpdateChecker::ClassifyFailure(404) == UpdateChecker::Failure::Http);
    CHECK(UpdateChecker::ClassifyFailure(500) == UpdateChecker::Failure::Http);
    CHECK(UpdateChecker::ClassifyFailure(0) == UpdateChecker::Failure::Network);
    CHECK(UpdateChecker::ClassifyFailure(200) == UpdateChecker::Failure::Network);
}

TEST_CASE("Lichii does not select Eden or lookalike download URLs", "[updater]") {
    const std::string url = GENERATE(
        "https://github.com/eden-emu/eden/releases/download/build-71/lichii-windows-x86_64.zip",
        "https://github.com/zortos293/Lichii/releases/download/build-70/lichii-windows-x86_64.zip",
        "https://github.com/zortos293/Lichii/releases/download/build-71/"
        "not-lichii-windows-x86_64.zip",
        "https://github.com.example.com/zortos293/Lichii/releases/download/build-71/"
        "lichii-windows-x86_64.zip");
    auto release = MakeRelease("build-71", "lichii-windows-x86_64.zip");
    release.assets = {url};
    CHECK(release.GetPlatformAssets("windows-x86_64").empty());
}

TEST_CASE("GitHub failures and invalid JSON produce no releases", "[updater]") {
    const std::string_view json =
        GENERATE("", "not JSON", R"({"message":"API rate limit exceeded"})", "null", "[]",
                 R"([{"tag_name":42}])", R"([{"tag_name":"build-71","assets":null}])",
                 R"([{"tag_name":"build-71","draft":true}])");
    CHECK(Common::Net::Release::ListFromJson(json, "https://api.github.com", "zortos293/Lichii")
              .empty());
}

TEST_CASE("GitHub optional fields can be null or missing", "[updater]") {
    const std::string_view json = R"([{"tag_name":"build-71","name":null,"body":null}])";
    const auto releases =
        Common::Net::Release::ListFromJson(json, "https://api.github.com", "zortos293/Lichii");
    REQUIRE(releases.size() == 1);
    CHECK(releases[0].title == "build-71");
    CHECK(releases[0].assets.empty());
    CHECK(releases[0].html_url == "https://github.com/zortos293/Lichii/releases/tag/build-71");
}

TEST_CASE("Release requests handle HTTP failures without throwing", "[updater]") {
    const int status = GENERATE(200, 403, 429, 500);
    httplib::Server server;
    server.Get("/releases", [status](const httplib::Request& request, httplib::Response& response) {
        response.status =
            request.get_header_value("User-Agent") == "Lichii" &&
                    request.get_header_value("Accept") == "application/vnd.github+json"
                ? status
                : 400;
        response.set_content("[]", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    const std::jthread listener([&server](std::stop_token stop) {
        const std::stop_callback stop_server(stop, [&server] { server.stop(); });
        server.listen_after_bind();
    });
    server.wait_until_ready();
    const auto response =
        Common::Net::MakeRequest("http://127.0.0.1:" + std::to_string(port), "/releases");
    CHECK(response.has_value() == (status == 200));
}

namespace {

using UpdateChecker::FritchiiVersion;

const std::string ValidDigest =
    "sha256:597200A1BFFA3D2ECFB2F3F2A464515D57D7F6DC674060F16719830C790BADA7";

Common::Net::Release MakeFritchiiRelease(const std::string& tag,
                                         const std::vector<std::string>& urls,
                                         const std::string& digest = ValidDigest) {
    nlohmann::json assets = nlohmann::json::array();
    for (const auto& url : urls) {
        assets.push_back(
            {{"browser_download_url", url}, {"size", u64{68126892}}, {"digest", digest}});
    }
    const nlohmann::json json{
        {"tag_name", tag}, {"name", "Fritchii " + tag}, {"prerelease", true}, {"assets", assets}};
    return Common::Net::Release::FromJson(json, "https://api.github.com", "Frizti/Lichii").value();
}

std::string GitHubUrl(std::string_view repo, std::string_view tag, std::string_view name) {
    return fmt::format("https://github.com/Frizti/{}/releases/download/{}/{}", repo, tag, name);
}

} // namespace

TEST_CASE("Fritchii versions compare numerically, with a patch number", "[updater]") {
    CHECK(UpdateChecker::ParseFritchiiVersion("4") == FritchiiVersion{4, 0});
    CHECK(UpdateChecker::ParseFritchiiVersion("3.5") == FritchiiVersion{3, 5});
    CHECK(UpdateChecker::ParseFritchiiVersion("12") == FritchiiVersion{12, 0});
    CHECK(UpdateChecker::ParseFritchiiVersion("5.10") == FritchiiVersion{5, 10});
    CHECK(UpdateChecker::ParseFritchiiVersion("5.6.1") == FritchiiVersion{5, 6, 1});
    CHECK(UpdateChecker::ParseFritchiiVersion("5.0.1") == FritchiiVersion{5, 0, 1});
    CHECK(FritchiiVersion{5, 9} < FritchiiVersion{5, 10});
    CHECK(FritchiiVersion{5, 6} < FritchiiVersion{5, 6, 1});
    CHECK(FritchiiVersion{5, 6, 1} < FritchiiVersion{5, 7});
    CHECK(FritchiiVersion{5, 6, 1}.ToString() == "5.6.1");
    CHECK(FritchiiVersion{3, 5} < FritchiiVersion{4, 0});
    CHECK(FritchiiVersion{4, 0} < FritchiiVersion{4, 1});
    CHECK(FritchiiVersion{4, 9} < FritchiiVersion{5, 0});
    CHECK(FritchiiVersion{9, 0} < FritchiiVersion{10, 0});
    CHECK(FritchiiVersion{4, 1}.ToString() == "4.1");
    CHECK(FritchiiVersion{4, 0}.ToString() == "4");

    const std::string invalid = GENERATE("", "0", "04", "4.", "4.0", "4.01", "4.a", "v4", " 4",
                                         "4 ", "-1", "+4", "4.1.", "4.1.0", "4.1.02", "4.1.2.3",
                                         ".5", "99999999999");
    CHECK_FALSE(UpdateChecker::ParseFritchiiVersion(invalid));
}

TEST_CASE("Fritchii tags and build versions accept the old Frizti names", "[updater]") {
    CHECK(UpdateChecker::ParseFritchiiTag("fritchii-v4") == FritchiiVersion{4, 0});
    CHECK(UpdateChecker::ParseFritchiiTag("fritchii-v4.1") == FritchiiVersion{4, 1});
    CHECK(UpdateChecker::ParseFritchiiTag("frizti-v3") == FritchiiVersion{3, 0});
    CHECK(UpdateChecker::ParseFritchiiTag("frizti-v3.5") == FritchiiVersion{3, 5});
    CHECK(UpdateChecker::ParseBuildVersion("Fritchii v4") == FritchiiVersion{4, 0});
    CHECK(UpdateChecker::ParseBuildVersion("Frizti v3.5") == FritchiiVersion{3, 5});

    const std::string tag =
        GENERATE("", "fritchii-v", "fritchii-4", "Fritchii-v4", "fritchii-v4-rc", "frizti-v4.0",
                 "build-71", "v4", "lichii-frizti-v4");
    CHECK_FALSE(UpdateChecker::ParseFritchiiTag(tag));
    const std::string version = GENERATE("", "build-71", "Lichii", "abc1234567-nx2", "Fritchii v",
                                         "Fritchii v4 ", "fritchii v4", "Fritchii 4");
    CHECK_FALSE(UpdateChecker::ParseBuildVersion(version));
}

TEST_CASE("GitHub digests are parsed only when they are SHA-256", "[updater]") {
    CHECK(Common::Net::ParseSha256Digest(ValidDigest) ==
          "597200a1bffa3d2ecfb2f3f2a464515d57d7f6dc674060f16719830c790bada7");
    const std::string digest =
        GENERATE(std::string{}, std::string{"sha256:"}, "sha256:" + std::string(63, 'a'),
                 "sha256:" + std::string(65, 'a'), "sha256:" + std::string(63, 'a') + "g",
                 "SHA256:" + std::string(64, 'a'), "md5:" + std::string(64, 'a'),
                 " sha256:" + std::string(64, 'a'), std::string(64, 'a'));
    CHECK_FALSE(Common::Net::ParseSha256Digest(digest));
}

TEST_CASE("Fritchii selects only the Windows package from the Frizti repository", "[updater]") {
    const auto source = UpdateChecker::DefaultSource();
    CHECK(source.origin == "https://api.github.com");
    CHECK(source.api_path == "/repos/Frizti/Lichii/releases?per_page=100");
    CHECK_FALSE(source.test);

    const std::string tag = "fritchii-v5";
    const std::string repo = GENERATE("Lichii", "Fritchii");
    const auto release = MakeFritchiiRelease(
        tag, {GitHubUrl(repo, tag, "Fritchii-v5-debug-symbols-not-needed-to-play.zip"),
              GitHubUrl(repo, tag, "Fritchii-v5-Windows-x64.zip")});
    const auto asset = UpdateChecker::SelectFritchiiAsset(release, source);
    REQUIRE(asset);
    CHECK(asset->filename == "Fritchii-v5-Windows-x64.zip");
    CHECK(asset->url == "https://github.com");
    CHECK(asset->path == "/Frizti/" + repo + "/releases/download/fritchii-v5/" + asset->filename);
    CHECK(asset->sha256 == "597200a1bffa3d2ecfb2f3f2a464515d57d7f6dc674060f16719830c790bada7");
    CHECK(asset->size == 68126892);

    // Selected, but the dialog refuses to install it without a digest.
    const auto no_digest = MakeFritchiiRelease(
        tag, {GitHubUrl(repo, tag, "Fritchii-v5-Windows-x64.zip")}, "sha256:short");
    const auto unverified = UpdateChecker::SelectFritchiiAsset(no_digest, source);
    REQUIRE(unverified);
    CHECK(unverified->sha256.empty());
}

TEST_CASE("Fritchii ignores debug zips, old asset names and lookalike URLs", "[updater]") {
    const std::string url = GENERATE(
        GitHubUrl("Lichii", "fritchii-v5", "Fritchii-v5-debug-symbols-not-needed-to-play.zip"),
        GitHubUrl("Lichii", "fritchii-v5", "Lichii-Frizti-v5-Windows-x64.zip"),
        GitHubUrl("Lichii", "fritchii-v5", "Fritchii-v4-Windows-x64.zip"),
        GitHubUrl("Lichii", "fritchii-v4", "Fritchii-v5-Windows-x64.zip"),
        GitHubUrl("Other", "fritchii-v5", "Fritchii-v5-Windows-x64.zip"),
        std::string{"https://github.com/zortos293/Lichii/releases/download/fritchii-v5/"
                    "Fritchii-v5-Windows-x64.zip"},
        std::string{"http://github.com/Frizti/Lichii/releases/download/fritchii-v5/"
                    "Fritchii-v5-Windows-x64.zip"},
        std::string{"https://github.com.evil.com/Frizti/Lichii/releases/download/fritchii-v5/"
                    "Fritchii-v5-Windows-x64.zip"},
        std::string{"http://127.0.0.1:8000/Fritchii-v5-Windows-x64.zip"},
        std::string{"https://github.com/Frizti/Lichii/archive/refs/tags/fritchii-v5.zip"});
    const auto release = MakeFritchiiRelease("fritchii-v5", {url});
    CHECK_FALSE(UpdateChecker::SelectFritchiiAsset(release, UpdateChecker::DefaultSource()));
}

TEST_CASE("Fritchii picks the newest newer release that has a package", "[updater]") {
    const auto make = [](const std::string& tag, const std::string& name) {
        return MakeFritchiiRelease(tag, {GitHubUrl("Lichii", tag, name)});
    };
    const auto releases = std::array{
        make("frizti-v3", "Lichii-Frizti-v3-Windows-x64.zip"),
        make("fritchii-v4", "Fritchii-v4-Windows-x64.zip"),
        make("fritchii-v5", "Fritchii-v5-debug-symbols-not-needed-to-play.zip"),
        make("fritchii-v4.1", "Fritchii-v4.1-Windows-x64.zip"),
        make("build-99", "lichii-windows-x86_64.zip"),
    };
    const auto source = UpdateChecker::DefaultSource();
    const auto select = [&](FritchiiVersion current) {
        const auto update = UpdateChecker::SelectFritchiiUpdate(releases, current, source);
        return update ? update->tag : std::string{};
    };
    CHECK(select({3, 5}) == "fritchii-v4.1");
    CHECK(select({4, 0}) == "fritchii-v4.1");
    CHECK(select({4, 1}).empty());
    CHECK(select({6, 0}).empty());
    CHECK_FALSE(UpdateChecker::SelectFritchiiUpdate({}, {4, 0}, source));
}

TEST_CASE("Fritchii never offers a release whose notes hide it from the updater", "[updater]") {
    const auto make = [](const std::string& tag, const std::string& name, std::string body) {
        auto release = MakeFritchiiRelease(tag, {GitHubUrl("Lichii", tag, name)});
        release.body = std::move(body);
        return release;
    };
    const auto releases = std::array{
        make("fritchii-v5.6", "Fritchii-v5.6-Windows-x64.zip", "Real notes"),
        make("fritchii-v9", "Fritchii-v9-Windows-x64.zip",
             "Big news!\n<!-- fritchii:no-update -->\nMore text"),
    };
    const auto source = UpdateChecker::DefaultSource();
    CHECK(UpdateChecker::HiddenFromUpdater(releases[1]));
    CHECK_FALSE(UpdateChecker::HiddenFromUpdater(releases[0]));
    const auto update = UpdateChecker::SelectFritchiiUpdate(releases, {5, 5}, source);
    REQUIRE(update);
    CHECK(update->tag == "fritchii-v5.6");
    CHECK_FALSE(UpdateChecker::SelectFritchiiUpdate(releases, {5, 6}, source));
}

TEST_CASE("EDEN_UPDATE_URL takes https anywhere and plain http only on loopback", "[updater]") {
    const auto local = UpdateChecker::ParseSourceOverride("http://127.0.0.1:8000/releases.json");
    REQUIRE(local);
    CHECK(local->origin == "http://127.0.0.1:8000");
    CHECK(local->api_path == "/releases.json");
    CHECK(local->test);
    CHECK(UpdateChecker::ParseSourceOverride("http://localhost:8000")->api_path == "/");
    CHECK(UpdateChecker::ParseSourceOverride("http://LocalHost/x.json"));
    CHECK(UpdateChecker::ParseSourceOverride("http://[::1]:8000/x.json")->origin ==
          "http://[::1]:8000");
    CHECK(UpdateChecker::ParseSourceOverride("https://example.com/r.json")->origin ==
          "https://example.com");

    const std::string url =
        GENERATE("", "127.0.0.1:8000/r.json", "http://example.com/r.json",
                 "http://127.0.0.1.evil.com/r.json", "http://10.0.0.1/r.json",
                 "http://localhost@evil.com/r.json", "ftp://127.0.0.1/r.json", "file:///C:/r.json",
                 "http://127.0.0.1 /r.json", "http:///r.json", "https://");
    CHECK_FALSE(UpdateChecker::ParseSourceOverride(url));
}

TEST_CASE("A test source accepts packages only from its own origin", "[updater]") {
    const auto source = UpdateChecker::ParseSourceOverride("http://127.0.0.1:8000/releases.json");
    REQUIRE(source);
    const auto local =
        MakeFritchiiRelease("fritchii-v5", {"http://127.0.0.1:8000/Fritchii-v5-Windows-x64.zip"});
    const auto asset = UpdateChecker::SelectFritchiiAsset(local, *source);
    REQUIRE(asset);
    CHECK(asset->url == "http://127.0.0.1:8000");
    CHECK(asset->path == "/Fritchii-v5-Windows-x64.zip");

    const std::string url =
        GENERATE(std::string{"http://127.0.0.1:9000/Fritchii-v5-Windows-x64.zip"},
                 std::string{"http://127.0.0.1:80000/Fritchii-v5-Windows-x64.zip"},
                 std::string{"http://127.0.0.1:8000/../Fritchii-v5-Windows-x64.zip"},
                 std::string{"http://127.0.0.1:8000/x?/Fritchii-v5-Windows-x64.zip"},
                 GitHubUrl("Lichii", "fritchii-v5", "Fritchii-v5-Windows-x64.zip"));
    CHECK_FALSE(
        UpdateChecker::SelectFritchiiAsset(MakeFritchiiRelease("fritchii-v5", {url}), *source));
}
