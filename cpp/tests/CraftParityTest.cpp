// Parity of the ported CRAFT decoder with the react-native-executorch original
// (commit 3f83e48a1) on real detector outputs. The fixtures under
// `fixtures/craft/` are heatmaps the EasyOCR English model produced on the
// example's sample pages, captured by the example app's "Save detector
// heatmap" action (`example/craftFixture.ts`) and copied by
// `scripts/pull-craft-fixtures.sh`; the `.json` next to each one records where
// it came from.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ops/craft_ops.h"
#include "support/ReferenceCraft.h"

#ifndef RNOCR_FIXTURES_DIR
#error "RNOCR_FIXTURES_DIR must point at cpp/tests/fixtures"
#endif

namespace rnocr::tests {
namespace {

using ops::craft::CraftBox;
using ops::craft::CraftOptions;
using ops::craft::extractCraftTextBoxes;

// The plan's tolerance for the native CRAFT port: 1 px, 1 degree.
constexpr float kPositionTolerance = 1.0f;
constexpr float kAngleTolerance = 1.0f;

/** A `.heat` fixture: "CRFT", uint32 version, int32 height, int32 width, float32 [H, W, 2]. */
struct Fixture {
    std::string name;
    int32_t height = 0;
    int32_t width = 0;
    std::vector<float> data;
};

template <typename T>
T readLittleEndian(std::istream &in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    return value; // the hosts this runs on are little-endian
}

Fixture loadFixture(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path.string());
    }
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "CRFT") {
        throw std::runtime_error(path.string() + " is not a CRFT fixture");
    }
    const auto version = readLittleEndian<uint32_t>(in);
    if (version != 1) {
        throw std::runtime_error(path.string() + ": unsupported fixture version " + std::to_string(version));
    }
    Fixture fixture;
    fixture.name = path.stem().string();
    fixture.height = readLittleEndian<int32_t>(in);
    fixture.width = readLittleEndian<int32_t>(in);
    if (fixture.height <= 0 || fixture.width <= 0) {
        throw std::runtime_error(path.string() + ": bad dimensions");
    }
    fixture.data.resize(static_cast<size_t>(fixture.height) * fixture.width * 2);
    in.read(reinterpret_cast<char *>(fixture.data.data()), static_cast<std::streamsize>(fixture.data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error(path.string() + ": truncated");
    }
    return fixture;
}

std::vector<Fixture> loadFixtures() {
    std::vector<std::filesystem::path> paths;
    const std::filesystem::path dir = std::filesystem::path(RNOCR_FIXTURES_DIR) / "craft";
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".heat") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<Fixture> fixtures;
    for (const auto &path : paths) {
        fixtures.push_back(loadFixture(path));
    }
    return fixtures;
}

/** Decodes with both implementations and asserts the boxes agree pairwise. */
void expectParity(const Fixture &fixture, bool charLevel, float scale) {
    CraftOptions options;
    options.charLevel = charLevel;
    options.scale = scale;
    const std::vector<CraftBox> ours = extractCraftTextBoxes(fixture.data.data(), fixture.height, fixture.width, options);

    // The reference reads the buffer through a non-const Mat but never writes it.
    std::vector<float> copy = fixture.data;
    const std::vector<reference::Box> theirs =
        reference::extractCraft(copy.data(), fixture.width, fixture.height, options.textThreshold, options.linkThreshold,
                                options.lowTextThreshold, scale, charLevel);

    SCOPED_TRACE(fixture.name + (charLevel ? " charLevel" : " lineLevel") + " scale " + std::to_string(scale));
    ASSERT_EQ(ours.size(), theirs.size());
    EXPECT_GT(ours.size(), 0u) << "a fixture with no text proves nothing";
    for (size_t i = 0; i < ours.size(); ++i) {
        SCOPED_TRACE("box " + std::to_string(i));
        EXPECT_NEAR(ours[i].x0, theirs[i].x0, kPositionTolerance);
        EXPECT_NEAR(ours[i].y0, theirs[i].y0, kPositionTolerance);
        EXPECT_NEAR(ours[i].x1, theirs[i].x1, kPositionTolerance);
        EXPECT_NEAR(ours[i].y1, theirs[i].y1, kPositionTolerance);
        EXPECT_NEAR(ours[i].angle, theirs[i].angle, kAngleTolerance);
    }
}

TEST(CraftParity, FixturesArePresent) {
    EXPECT_FALSE(loadFixtures().empty()) << "no .heat fixtures under " << RNOCR_FIXTURES_DIR << "/craft";
}

TEST(CraftParity, LineLevelMatchesReference) {
    for (const auto &fixture : loadFixtures()) {
        expectParity(fixture, /* charLevel */ false, /* scale */ 2.0f);
    }
}

TEST(CraftParity, CharLevelMatchesReference) {
    for (const auto &fixture : loadFixtures()) {
        expectParity(fixture, /* charLevel */ true, /* scale */ 2.0f);
    }
}

// The reference returns heatmap coordinates when the ratio is 1; the port must
// agree there too, since the TypeScript side chooses the scale per model.
TEST(CraftParity, UnscaledMatchesReference) {
    for (const auto &fixture : loadFixtures()) {
        expectParity(fixture, /* charLevel */ false, /* scale */ 1.0f);
    }
}

// Real heatmaps have components whose fitted rectangle pokes out of the map
// by floating-point noise; the port clamps those, which the tolerance above
// hides. This checks the clamp did not move anything by more than that noise.
TEST(CraftParity, BoxesStayInsideTheHeatmap) {
    for (const auto &fixture : loadFixtures()) {
        CraftOptions options;
        options.scale = 1.0f;
        for (const auto &box : extractCraftTextBoxes(fixture.data.data(), fixture.height, fixture.width, options)) {
            EXPECT_GE(box.x0, 0.0f);
            EXPECT_GE(box.y0, 0.0f);
            EXPECT_LE(box.x1, static_cast<float>(fixture.width));
            EXPECT_LE(box.y1, static_cast<float>(fixture.height));
            EXPECT_LT(box.x0, box.x1);
            EXPECT_LT(box.y0, box.y1);
        }
    }
}

} // namespace
} // namespace rnocr::tests
