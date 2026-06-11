import Foundation

struct VideoItem: Identifiable, Codable {
    let id: UUID
    let jobId: String
    let outputPath: String
    let prompt: String
    let seed: Int
    let width: Int
    let height: Int
    let numFrames: Int
    let fps: Int
    let durationSeconds: Double
    let generationTimeSeconds: Double?
    let createdAt: Date
    let generationType: String

    var fileURL: URL { URL(fileURLWithPath: outputPath) }
    var displayName: String { "Video \(jobId)" }
    var resolutionLabel: String { "\(width)×\(height)" }

    /// "6m 47s" style label for the wall-clock generation time, if recorded.
    var generationTimeLabel: String? {
        guard let t = generationTimeSeconds, t > 0 else { return nil }
        let total = Int(t.rounded())
        return total >= 60 ? "\(total / 60)m \(total % 60)s" : "\(total)s"
    }

    var generationTypeLabel: String {
        switch generationType {
        case "t2v": return "Text to Video"
        case "i2v": return "Image to Video"
        case "preview": return "Preview"
        default: return generationType.uppercased()
        }
    }

    /// Decode from the backend history API JSON (snake_case keys).
    enum CodingKeys: String, CodingKey {
        case id
        case jobId = "job_id"
        case outputPath = "output_path"
        case prompt
        case seed
        case width, height
        case numFrames = "num_frames"
        case fps
        case durationSeconds = "duration_seconds"
        case generationTimeSeconds = "generation_time_seconds"
        case createdAt = "created_at"
        case generationType = "generation_type"
    }
}
