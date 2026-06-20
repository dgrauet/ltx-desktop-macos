import Foundation
import CoreML

enum CoreMLModelError: Error { case downloadFailed, compileFailed }

/// Downloads + compiles + caches the Depth-Anything-V2-Small Core ML model on demand.
actor CoreMLModelManager {
    static let shared = CoreMLModelManager()

    private var cachedDepth: MLModel?

    private static let repo = "apple/coreml-depth-anything-v2-small"
    private static let pkg = "DepthAnythingV2SmallF16.mlpackage"
    private static let revision = "cfef6f6f2a70783dedc0bfae40cecbc2052285d3"
    private static let files = [
        "Manifest.json",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin",
    ]

    private var supportDir: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        return base.appendingPathComponent("LTXDesktop/coreml", isDirectory: true)
    }

    func depthModel(progress: @escaping (Double) -> Void) async throws -> MLModel {
        if let m = cachedDepth { return m }

        let pkgDir = supportDir.appendingPathComponent(Self.pkg, isDirectory: true)
        let compiledURL = supportDir.appendingPathComponent("DepthAnythingV2SmallF16.mlmodelc")

        // Compile once and cache the .mlmodelc.
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            if !FileManager.default.fileExists(atPath: pkgDir.path) {
                do {
                    try await downloadPackage(into: pkgDir, progress: progress)
                } catch {
                    try? FileManager.default.removeItem(at: pkgDir)
                    throw error
                }
            }
            progress(0.95)
            let compiled: URL
            do {
                compiled = try await MLModel.compileModel(at: pkgDir)
            } catch { throw CoreMLModelError.compileFailed }
            try? FileManager.default.removeItem(at: compiledURL)
            try FileManager.default.moveItem(at: compiled, to: compiledURL)
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        cachedDepth = model
        return model
    }

    private func downloadPackage(into pkgDir: URL,
                                 progress: @escaping (Double) -> Void) async throws {
        try FileManager.default.createDirectory(at: pkgDir, withIntermediateDirectories: true)
        let total = Double(Self.files.count)
        for (i, rel) in Self.files.enumerated() {
            let urlStr = "https://huggingface.co/\(Self.repo)/resolve/\(Self.revision)/"
                + Self.pkg + "/" + rel
            guard let url = URL(string: urlStr) else { throw CoreMLModelError.downloadFailed }
            let dest = pkgDir.appendingPathComponent(rel)
            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            do {
                let (tmp, response) = try await URLSession.shared.download(from: url)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw CoreMLModelError.downloadFailed
                }
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
            } catch { throw CoreMLModelError.downloadFailed }
            progress(Double(i + 1) / total)
        }
    }
}
