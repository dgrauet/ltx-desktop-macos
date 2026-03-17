import Foundation

/// A saved parameter preset for video generation.
struct GenerationPreset: Codable, Identifiable {
    let id: String
    let name: String
    let builtin: Bool
    let createdAt: String
    let params: PresetParams

    enum CodingKeys: String, CodingKey {
        case id, name, builtin, params
        case createdAt = "created_at"
    }
}

/// The generation parameters stored in a preset.
struct PresetParams: Codable {
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int
    let seed: Int
    let fps: Int
    let guidanceScale: Double?
    let negativePrompt: String?
    let generateAudio: Bool?
    let ffmpegUpscale: Bool?

    enum CodingKeys: String, CodingKey {
        case width, height, steps, seed, fps
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case negativePrompt = "negative_prompt"
        case generateAudio = "generate_audio"
        case ffmpegUpscale = "ffmpeg_upscale"
    }
}

/// Request body for creating a new preset.
struct CreatePresetRequest: Encodable {
    let name: String
    let params: PresetParams
}
