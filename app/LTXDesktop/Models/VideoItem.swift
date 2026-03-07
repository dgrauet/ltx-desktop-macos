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
    let createdAt: Date

    var fileURL: URL { URL(fileURLWithPath: outputPath) }
    var displayName: String { "Video \(jobId)" }
    var resolutionLabel: String { "\(width)×\(height)" }
}
