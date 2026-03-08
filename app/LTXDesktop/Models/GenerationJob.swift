import Foundation

struct T2VRequest: Codable {
    var prompt: String
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 8
    var seed: Int = 42
    var guidanceScale: Double = 1.0
    var fps: Int = 24

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
    }
}

struct PreviewRequest: Codable {
    var prompt: String
    var seed: Int = 42
    var fps: Int = 24
    var sourceImagePath: String?
    var imageStrength: Double = 1.0

    enum CodingKeys: String, CodingKey {
        case prompt, seed, fps
        case sourceImagePath = "source_image_path"
        case imageStrength = "image_strength"
    }
}

struct I2VRequest: Codable {
    var prompt: String
    var sourceImagePath: String
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 8
    var seed: Int = 42
    var guidanceScale: Double = 1.0
    var fps: Int = 24
    var imageStrength: Double = 0.85

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case sourceImagePath = "source_image_path"
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case imageStrength = "image_strength"
    }
}

struct JobResponse: Codable {
    let jobId: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
    }
}

struct JobStatus: Codable {
    let status: String
    let progress: Double
    let result: JobResult?
    let error: String?
}

struct JobResult: Codable {
    let jobId: String
    let outputPath: String
    let durationSeconds: Double
    let stages: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case outputPath = "output_path"
        case durationSeconds = "duration_seconds"
        case stages
    }
}

struct CancelResponse: Codable {
    let success: Bool
}

struct ProgressUpdate: Codable {
    let jobId: String?
    let step: Int?
    let totalSteps: Int?
    let pct: Double?
    let done: Bool?
    let error: String?
    let previewFrame: String?

    enum CodingKeys: String, CodingKey {
        case step, pct, done, error
        case jobId = "job_id"
        case totalSteps = "total_steps"
        case previewFrame = "preview_frame"
    }
}
