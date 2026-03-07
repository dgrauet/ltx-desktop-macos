import Foundation

struct ModelInfo: Identifiable, Codable {
    let id: String
    let name: String
    let sizeGb: Double
    let loaded: Bool

    var sizeLabel: String { String(format: "%.1f GB", sizeGb) }

    enum CodingKeys: String, CodingKey {
        case id, name, loaded
        case sizeGb = "size_gb"
    }
}
