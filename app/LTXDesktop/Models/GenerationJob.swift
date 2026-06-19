import Foundation

struct T2VRequest: Codable {
    var prompt: String
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 8
    var seed: Int = -1
    var guidanceScale: Double = 3.0
    var fps: Int = 24
    var pipelineType: String = "distilled"
    var lowRam: Bool = false
    var loraIds: [String] = []

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case pipelineType = "pipeline_type"
        case lowRam = "low_ram"
        case loraIds = "lora_ids"
    }
}


struct I2VRequest: Codable {
    var prompt: String
    var sourceImagePath: String
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 8
    var seed: Int = -1
    var guidanceScale: Double = 3.0
    var fps: Int = 24
    var pipelineType: String = "distilled"
    var lowRam: Bool = false
    var imageStrength: Double = 1.0
    var loraIds: [String] = []

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case sourceImagePath = "source_image_path"
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case pipelineType = "pipeline_type"
        case lowRam = "low_ram"
        case imageStrength = "image_strength"
        case loraIds = "lora_ids"
    }
}

struct A2VRequest: Codable {
    var prompt: String
    var sourceAudioPath: String
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 30
    var seed: Int = -1
    var guidanceScale: Double = 3.0
    var fps: Int = 24
    var audioStart: Double = 0.0
    var lowRam: Bool = false
    var loraIds: [String] = []

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case sourceAudioPath = "source_audio_path"
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case audioStart = "audio_start"
        case lowRam = "low_ram"
        case loraIds = "lora_ids"
    }
}

struct ICLoraRequest: Codable {
    var prompt: String
    var sourceControlPath: String
    var extractEdges: Bool = false
    var icLoraId: String? = nil
    var icLoraPath: String? = nil
    var icLoraStrength: Double = 1.0
    var controlStrength: Double = 1.0
    var conditioningStrength: Double = 1.0
    var width: Int = 768
    var height: Int = 512
    var numFrames: Int = 97
    var steps: Int = 30
    var seed: Int = -1
    var guidanceScale: Double = 3.0
    var fps: Int = 24
    var skipStage2: Bool = false
    var lowRam: Bool = false

    enum CodingKeys: String, CodingKey {
        case prompt, width, height, steps, seed, fps
        case sourceControlPath = "source_control_path"
        case extractEdges = "extract_edges"
        case icLoraId = "ic_lora_id"
        case icLoraPath = "ic_lora_path"
        case icLoraStrength = "ic_lora_strength"
        case controlStrength = "control_strength"
        case conditioningStrength = "conditioning_strength"
        case numFrames = "num_frames"
        case guidanceScale = "guidance_scale"
        case skipStage2 = "skip_stage_2"
        case lowRam = "low_ram"
    }
}

struct JobResponse: Codable {
    let jobId: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
    }
}

/// Response after submitting a job to the priority queue.
struct QueueSubmitResponse: Codable {
    let jobId: String
    let position: Int
    let queueLength: Int

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case position
        case queueLength = "queue_length"
    }
}

/// A single entry from GET /api/v1/queue.
struct QueueResponse: Codable {
    let jobs: [QueueEntry]
    let paused: Bool
}

struct QueueEntry: Codable, Identifiable, Equatable {
    let jobId: String
    let jobType: String
    let priority: String
    let state: String
    let position: Int
    let prompt: String
    let submittedAt: Double
    let etaSeconds: Double?
    var progress: Double?
    var status: String?

    var id: String { jobId }

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case jobType = "job_type"
        case priority, state, position, prompt, progress, status
        case submittedAt = "submitted_at"
        case etaSeconds = "eta_seconds"
    }
}

struct JobStatus: Codable {
    let status: String
    let progress: Double
    let result: JobResult?
    let error: String?
    let position: Int?
    let etaSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case status, progress, result, error, position
        case etaSeconds = "eta_seconds"
    }
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
    let status: String?
    let queueLength: Int?
    let position: Int?
    let etaSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case step, pct, done, error, status, position
        case jobId = "job_id"
        case totalSteps = "total_steps"
        case previewFrame = "preview_frame"
        case queueLength = "queue_length"
        case etaSeconds = "eta_seconds"
    }
}
