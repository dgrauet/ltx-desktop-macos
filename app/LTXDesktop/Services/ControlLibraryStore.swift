import Foundation
import AVFoundation
import AppKit

/// Persists pose/depth control videos to ~/.ltx-desktop/control-videos/ with a
/// thumbnail and a JSON index. Dedup by (source, type, dims, frames).
actor ControlLibraryStore {
    static let shared = ControlLibraryStore()

    private var dir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ltx-desktop/control-videos", isDirectory: true)
    }
    private var indexURL: URL { dir.appendingPathComponent("library.json") }

    static func dedupKey(sourceName: String, controlType: String,
                         width: Int, height: Int, frameCount: Int) -> String {
        "\(sourceName)|\(controlType)|\(width)x\(height)|\(frameCount)"
    }

    func list() -> [ControlLibraryItem] {
        guard let data = try? Data(contentsOf: indexURL),
              let items = try? JSONDecoder.iso.decode([ControlLibraryItem].self, from: data)
        else { return [] }
        return items.sorted { $0.createdAt > $1.createdAt }
    }

    func delete(id: String) {
        var items = list()
        guard let item = items.first(where: { $0.id == id }) else { return }
        try? FileManager.default.removeItem(atPath: item.videoPath)
        try? FileManager.default.removeItem(atPath: item.thumbnailPath)
        items.removeAll { $0.id == id }
        try? persist(items)
    }

    func save(videoURL: URL, controlType: String, sourceName: String) async throws -> ControlLibraryItem {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let asset = AVURLAsset(url: videoURL)
        let track = try await asset.loadTracks(withMediaType: .video).first
        let size = try await track?.load(.naturalSize) ?? .zero
        let width = Int(abs(size.width)), height = Int(abs(size.height))
        let frameCount = try await Self.countFrames(asset: asset)
        let key = Self.dedupKey(sourceName: sourceName, controlType: controlType,
                                width: width, height: height, frameCount: frameCount)

        var items = list()
        // Dedup: if same key exists, remove its files + entry, reuse a fresh id.
        if let existing = items.first(where: { $0.dedupKey == key }) {
            try? FileManager.default.removeItem(atPath: existing.videoPath)
            try? FileManager.default.removeItem(atPath: existing.thumbnailPath)
            items.removeAll { $0.dedupKey == key }
        }

        let id = UUID().uuidString
        let destVideo = dir.appendingPathComponent("\(id).mp4")
        try? FileManager.default.removeItem(at: destVideo)
        try FileManager.default.copyItem(at: videoURL, to: destVideo)

        let thumbPath = dir.appendingPathComponent("\(id).png")
        try? await Self.writeThumbnail(asset: asset, to: thumbPath)

        let item = ControlLibraryItem(
            id: id, videoPath: destVideo.path, thumbnailPath: thumbPath.path,
            controlType: controlType, sourceName: sourceName,
            width: width, height: height, frameCount: frameCount,
            dedupKey: key, createdAt: Date())
        items.insert(item, at: 0)
        try persist(items)
        return item
    }

    private func persist(_ items: [ControlLibraryItem]) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONEncoder.iso.encode(items)
        try data.write(to: indexURL, options: .atomic)
    }

    private static func countFrames(asset: AVAsset) async throws -> Int {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { return 0 }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(out); reader.startReading()
        var n = 0
        while out.copyNextSampleBuffer() != nil { n += 1 }
        return n
    }

    private static func writeThumbnail(asset: AVAsset, to url: URL) async throws {
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        let cg = try await gen.image(at: .zero).image
        let rep = NSBitmapImageRep(cgImage: cg)
        guard let png = rep.representation(using: .png, properties: [:]) else { return }
        try png.write(to: url)
    }
}

private extension JSONEncoder {
    static var iso: JSONEncoder {
        let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e
    }
}
private extension JSONDecoder {
    static var iso: JSONDecoder {
        let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d
    }
}
