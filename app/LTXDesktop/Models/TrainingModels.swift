import Foundation

// MARK: - Dataset

struct TrainingDataset: Identifiable, Codable {
    let id: String
    let clipCount: Int
    let diskBytes: Int
    let hasPrecomputed: Bool

    enum CodingKeys: String, CodingKey {
        case id
        case clipCount = "clip_count"
        case diskBytes = "disk_bytes"
        case hasPrecomputed = "has_precomputed"
    }
}

// MARK: - Training Run

struct TrainingRun: Identifiable, Codable {
    let runId: String
    let datasetId: String
    let status: String
    let peakMemGb: Double?
    let loraPath: String?
    let createdAt: String

    var id: String { runId }

    enum CodingKeys: String, CodingKey {
        case runId = "run_id"
        case datasetId = "dataset_id"
        case status
        case peakMemGb = "peak_mem_gb"
        case loraPath = "lora_path"
        case createdAt = "created_at"
    }
}

// MARK: - Training Config Request

struct TrainingConfigRequest: Codable {
    let datasetId: String
    let lowRam: Bool
    let rank: Int
    let steps: Int
    let learningRate: Double?
    let seed: Int?

    enum CodingKeys: String, CodingKey {
        case datasetId = "dataset_id"
        case lowRam = "low_ram"
        case rank
        case steps
        case learningRate = "learning_rate"
        case seed
    }
}

// MARK: - Preflight Result

struct PreflightResult: Codable {
    let verdict: String
    let peakGb: Double

    enum CodingKeys: String, CodingKey {
        case verdict
        case peakGb = "peak_gb"
    }
}

// MARK: - Training WebSocket Events

/// Decoded from the training WebSocket stream.
/// Backend sends JSON dicts with a `type` discriminator.
/// - status:  {"type":"status","status":"<string>"}
/// - step:    {"type":"step","step":<int>,"loss":<float>,"lr":<float>,"peak_mem_gb":<float>}
/// - sample:  {"type":"sample","path":"<string>"}
/// - done:    {"type":"done","lora_path":"<string>"}
/// - error:   {"type":"error","message":"<string>"}
enum TrainingEvent: Decodable {
    case status(String)
    case step(step: Int, total: Int, peakMemGb: Double)
    case sample(String)
    case done(loraPath: String)
    case error(String)

    private enum CodingKeys: String, CodingKey {
        case type
        case status
        case step
        case total
        case peakMemGb = "peak_mem_gb"
        case path
        case loraPath = "lora_path"
        case message
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type_ = try container.decode(String.self, forKey: .type)
        switch type_ {
        case "status":
            let value = try container.decode(String.self, forKey: .status)
            self = .status(value)
        case "step":
            let stepNum = try container.decode(Int.self, forKey: .step)
            let total = (try? container.decode(Int.self, forKey: .total)) ?? 0
            let peakMem = try container.decode(Double.self, forKey: .peakMemGb)
            self = .step(step: stepNum, total: total, peakMemGb: peakMem)
        case "sample":
            let path = try container.decode(String.self, forKey: .path)
            self = .sample(path)
        case "done":
            let loraPath = try container.decode(String.self, forKey: .loraPath)
            self = .done(loraPath: loraPath)
        case "error":
            let message = try container.decode(String.self, forKey: .message)
            self = .error(message)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown training event type: \(type_)"
            )
        }
    }
}
