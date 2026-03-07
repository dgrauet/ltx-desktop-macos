import Foundation

struct ExportVideoRequest: Encodable {
    let videoPath: String
    let codec: String
    let outputFormat: String
    let bitrate: String

    enum CodingKeys: String, CodingKey {
        case videoPath = "video_path"
        case codec
        case outputFormat = "output_format"
        case bitrate
    }
}

struct ExportFCPXMLRequest: Encodable {
    let videoPath: String
    let clipName: String
    let frameRate: String

    enum CodingKeys: String, CodingKey {
        case videoPath = "video_path"
        case clipName = "clip_name"
        case frameRate = "frame_rate"
    }
}

struct PathResponse: Decodable {
    let outputPath: String

    enum CodingKeys: String, CodingKey {
        case outputPath = "output_path"
    }
}

struct LoadLoRARequest: Encodable {
    let loraId: String

    enum CodingKeys: String, CodingKey {
        case loraId = "lora_id"
    }
}
