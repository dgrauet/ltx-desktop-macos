import Foundation

struct ModelInfo: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let sizeGb: Double
    let modelType: String
    let downloaded: Bool
    let hfRepo: String

    var sizeLabel: String { String(format: "%.1f GB", sizeGb) }

    var typeLabel: String {
        switch modelType {
        case "video_generator": return "Video Generator"
        case "text_encoder": return "Text Encoder"
        case "prompt_enhancer": return "Prompt Enhancer"
        case "upscaler": return "Upscaler"
        default: return modelType
        }
    }

    enum CodingKeys: String, CodingKey {
        case id, name, description, downloaded
        case sizeGb = "size_gb"
        case modelType = "model_type"
        case hfRepo = "hf_repo"
    }
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
