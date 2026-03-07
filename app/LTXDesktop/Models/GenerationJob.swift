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

    enum CodingKeys: String, CodingKey {
        case step, pct, done, error
        case jobId = "job_id"
        case totalSteps = "total_steps"
    }
}
