import Foundation

struct ModelInfo: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let sizeGb: Double
    let modelType: String
    let downloaded: Bool
    let hfRepo: String
    /// "hf" (catalog, HF cache) or "local" (user-registered pack directory)
    let source: String?
    let gated: Bool?
    /// "2.3" / "2.5" for video generators
    let family: String?
    let capabilities: ModelCapabilities?

    var isLocal: Bool { source == "local" }
    var isLTX25: Bool { family == "2.5" }

    var sizeLabel: String { String(format: "%.1f GB", sizeGb) }

    var typeLabel: String {
        switch modelType {
        case "video_generator": return "Video Generator"
        case "text_encoder": return "Text Encoder"
        case "upscaler": return "Upscaler"
        case "ic-lora": return "IC-LoRA"
        default: return modelType
        }
    }

    enum CodingKeys: String, CodingKey {
        case id, name, description, downloaded, source, gated, family, capabilities
        case sizeGb = "size_gb"
        case modelType = "model_type"
        case hfRepo = "hf_repo"
    }
}

/// Per-family feature availability reported by the backend.
struct ModelCapabilities: Codable {
    let enhance: Bool
    let icLora: Bool
    let training: Bool
    let autoDuration: Bool
    let generatedKeyframes: Bool
    let diffusionDecoder: Bool

    enum CodingKeys: String, CodingKey {
        case enhance, training
        case icLora = "ic_lora"
        case autoDuration = "auto_duration"
        case generatedKeyframes = "generated_keyframes"
        case diffusionDecoder = "diffusion_decoder"
    }
}

struct LocalModelRequest: Encodable {
    let path: String
}

struct HFTokenStatus: Codable {
    let configured: Bool
    let user: String?
}

struct ModelListResponse: Codable {
    let models: [ModelInfo]
    let totalDiskGb: Double
    let selectedVideoModel: String?

    enum CodingKeys: String, CodingKey {
        case models
        case totalDiskGb = "total_disk_gb"
        case selectedVideoModel = "selected_video_model"
    }
}

struct ModelSelectRequest: Encodable {
    let modelId: String

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
    }
}

struct ModelDownloadRequest: Encodable {
    let modelId: String

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
    }
}

struct ModelDownloadResponse: Codable {
    let downloadId: String
    let modelId: String

    enum CodingKeys: String, CodingKey {
        case downloadId = "download_id"
        case modelId = "model_id"
    }
}

struct DownloadStatusResponse: Codable {
    let downloadId: String
    let modelId: String
    let status: String
    let progress: Double
    let error: String?

    enum CodingKeys: String, CodingKey {
        case downloadId = "download_id"
        case modelId = "model_id"
        case status, progress, error
    }
}

struct ModelDeleteResponse: Codable {
    let success: Bool
    let modelId: String
    let freedGb: Double
    let message: String

    enum CodingKeys: String, CodingKey {
        case success, message
        case modelId = "model_id"
        case freedGb = "freed_gb"
    }
}
