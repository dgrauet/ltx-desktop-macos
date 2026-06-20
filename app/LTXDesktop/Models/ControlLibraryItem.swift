import Foundation

struct ControlLibraryItem: Identifiable, Codable, Equatable {
    let id: String
    let videoPath: String
    let thumbnailPath: String
    let controlType: String   // "pose" | "depth"
    let sourceName: String
    let width: Int
    let height: Int
    let frameCount: Int
    let dedupKey: String
    let createdAt: Date

    var typeLabel: String {
        switch controlType {
        case "pose": return "Pose"
        case "depth": return "Depth"
        default: return controlType.capitalized
        }
    }
}
