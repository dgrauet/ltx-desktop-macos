// swift-tools-version:5.9
// Pure-Foundation editor core (document model, timeline operations, undo,
// persistence, FCPXML). The app target compiles these same sources directly;
// this package exists so the logic is unit-tested with `swift test`.
import PackageDescription

let package = Package(
    name: "LTXEditorCore",
    platforms: [.macOS(.v14)],
    products: [.library(name: "LTXEditorCore", targets: ["LTXEditorCore"])],
    targets: [
        .target(name: "LTXEditorCore"),
        .testTarget(name: "LTXEditorCoreTests", dependencies: ["LTXEditorCore"]),
    ]
)
