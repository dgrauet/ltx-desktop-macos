import SwiftUI
import AVFoundation

@MainActor
class HistoryViewModel: ObservableObject {
    @Published var videos: [VideoItem] = []
    @Published var selectedVideo: VideoItem? = nil
    @Published var thumbnails: [String: NSImage] = [:]
    @Published var isLoading = false
    @Published var errorMessage: String? = nil

    private let service = BackendService()

    private static let iso8601Formatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let iso8601FallbackFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    func loadVideos() {
        Task {
            isLoading = true
            errorMessage = nil
            do {
                let entries = try await service.fetchHistory()
                videos = entries.compactMap { entry in
                    // Parse ISO 8601 timestamp
                    let date = Self.iso8601Formatter.date(from: entry.createdAt)
                        ?? Self.iso8601FallbackFormatter.date(from: entry.createdAt)
                        ?? Date()

                    return VideoItem(
                        id: UUID(),
                        jobId: entry.jobId,
                        outputPath: entry.outputPath,
                        prompt: entry.prompt,
                        seed: entry.seed,
                        width: entry.width,
                        height: entry.height,
                        numFrames: entry.numFrames,
                        fps: entry.fps,
                        durationSeconds: entry.durationSeconds,
                        createdAt: date,
                        generationType: entry.generationType
                    )
                }
            } catch {
                // Fallback: scan local output files if backend is unavailable
                errorMessage = "Could not fetch history from backend. Showing local files."
                loadLocalVideos()
            }
            isLoading = false

            // Generate thumbnails for all loaded videos
            for item in videos {
                if thumbnails[item.jobId] == nil {
                    Task { await loadThumbnail(for: item) }
                }
            }
        }
    }

    func deleteVideo(_ item: VideoItem) {
        Task {
            do {
                try await service.deleteHistoryItem(jobId: item.jobId)
            } catch {
                // Even if backend delete fails, remove locally
            }
            // Also delete the video file from disk
            try? FileManager.default.removeItem(atPath: item.outputPath)
            videos.removeAll { $0.id == item.id }
            thumbnails.removeValue(forKey: item.jobId)
            if selectedVideo?.id == item.id { selectedVideo = nil }
        }
    }

    // MARK: - Fallback: Local File Scan

    private func loadLocalVideos() {
        let storageURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ltx-desktop/outputs")

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
                createdAt: created,
                generationType: "unknown"
            ))
        }
        videos = items.sorted { $0.createdAt > $1.createdAt }
    }

    // MARK: - Thumbnail Generation

    private func loadThumbnail(for item: VideoItem) async {
        let url = URL(fileURLWithPath: item.outputPath)
        guard FileManager.default.fileExists(atPath: item.outputPath) else { return }

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
