import SwiftUI
import AVFoundation

@MainActor
class HistoryViewModel: ObservableObject {
    @Published var videos: [VideoItem] = []
    @Published var selectedVideo: VideoItem? = nil
    @Published var thumbnails: [String: NSImage] = [:]

    private let storageURL: URL = {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ltx-desktop/outputs")
    }()

    func loadVideos() {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: storageURL,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        ) else {
            videos = []
            return
        }

        var items: [VideoItem] = []
        for fileURL in files where fileURL.pathExtension == "mp4" {
            let jsonURL = fileURL.deletingPathExtension().appendingPathExtension("json")
            if let data = try? Data(contentsOf: jsonURL),
               let item = try? JSONDecoder().decode(VideoItem.self, from: data) {
                items.append(item)
            } else {
                let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path)
                let created = attrs?[.creationDate] as? Date ?? Date()
                let jobId = fileURL.deletingPathExtension().lastPathComponent
                items.append(VideoItem(
                    id: UUID(),
                    jobId: jobId,
                    outputPath: fileURL.path,
                    prompt: "(no prompt)",
                    seed: 0,
                    width: 768,
                    height: 512,
                    numFrames: 0,
                    fps: 24,
                    durationSeconds: 0,
                    createdAt: created
                ))
            }
        }
        videos = items.sorted { $0.createdAt > $1.createdAt }

        for item in videos {
            Task { await loadThumbnail(for: item) }
        }
    }

    func deleteVideo(_ item: VideoItem) {
        try? FileManager.default.removeItem(atPath: item.outputPath)
        let jsonPath = item.outputPath.replacingOccurrences(of: ".mp4", with: ".json")
        try? FileManager.default.removeItem(atPath: jsonPath)
        videos.removeAll { $0.id == item.id }
        thumbnails.removeValue(forKey: item.jobId)
        if selectedVideo?.id == item.id { selectedVideo = nil }
    }

    private func loadThumbnail(for item: VideoItem) async {
        let url = URL(fileURLWithPath: item.outputPath)
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 320, height: 240)

        let time = CMTime(seconds: 0.1, preferredTimescale: 600)
        if let cgImage = try? await generator.image(at: time).image {
            thumbnails[item.jobId] = NSImage(cgImage: cgImage, size: .zero)
        }
    }
}
