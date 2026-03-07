import Foundation

struct LoRAInfo: Identifiable, Codable {
    let id: String
    let name: String
    let path: String
    let loraType: String
    let compatible: Bool
    var loaded: Bool
    let sizeMb: Double

    enum CodingKeys: String, CodingKey {
        case id, name, path, compatible, loaded
        case loraType = "lora_type"
        case sizeMb = "size_mb"
    }

    var typeLabel: String {
        switch loraType {
        case "camera_control": return "Camera Control"
        case "detail": return "Detail Enhancement"
        case "style": return "Style"
        default: return "Custom"
        }
    }

    var typeIcon: String {
        switch loraType {
        case "camera_control": return "camera.viewfinder"
        case "detail": return "sparkle.magnifyingglass"
        case "style": return "paintbrush"
        default: return "puzzlepiece"
        }
    }
}
