import Foundation

struct MemoryStats: Codable {
    let activeMemoryGb: Double
    let cacheMemoryGb: Double
    let peakMemoryGb: Double
    let systemAvailableGb: Double
    let generationCountSinceReload: Int
    let nextReloadIn: Int

    enum CodingKeys: String, CodingKey {
        case activeMemoryGb = "active_memory_gb"
        case cacheMemoryGb = "cache_memory_gb"
        case peakMemoryGb = "peak_memory_gb"
        case systemAvailableGb = "system_available_gb"
        case generationCountSinceReload = "generation_count_since_reload"
        case nextReloadIn = "next_reload_in"
    }
}

struct HealthResponse: Codable {
    let status: String
    let modelLoaded: Bool
    let generationCount: Int

    enum CodingKeys: String, CodingKey {
        case status
        case modelLoaded = "model_loaded"
        case generationCount = "generation_count"
    }
}

struct SystemInfoResponse: Codable {
    let chip: String
    let ramTotalGb: Int
    let ramAvailableGb: Double
    let macosVersion: String

    enum CodingKeys: String, CodingKey {
        case chip
        case ramTotalGb = "ram_total_gb"
        case ramAvailableGb = "ram_available_gb"
        case macosVersion = "macos_version"
    }
}

struct ResolutionLimit: Codable {
    let width: Int
    let height: Int
    let maxFrames: Int

    enum CodingKeys: String, CodingKey {
        case width, height
        case maxFrames = "max_frames"
    }
}

struct HardwareLimitsResponse: Codable {
    let totalRamGb: Int
    let maxResolution: ResolutionLimit
    let resolutionLimits: [ResolutionLimit]
    let warnings: [String]

    enum CodingKeys: String, CodingKey {
        case totalRamGb = "total_ram_gb"
        case maxResolution = "max_resolution"
        case resolutionLimits = "resolution_limits"
        case warnings
    }
}

/// Unified response for all memory-pressure endpoints.
///
/// Fields are optional where an endpoint may not return them:
/// - `pressureLevel`, `memory`: returned by GET /memory-pressure
/// - `resumed`: returned by POST /memory-pressure/resume
/// - `actionsTaken` / `lastActions`: mapped to the same property
struct MemoryPressureResponse: Codable {
    let pressureLevel: String?
    let actionsTaken: [String]?
    let pausedByPressure: Bool
    let autoPauseEnabled: Bool
    let autoCleanupEnabled: Bool
    let memory: MemoryStats?
    let resumed: Bool?

    enum CodingKeys: String, CodingKey {
        case pressureLevel = "pressure_level"
        case actionsTaken = "actions_taken"
        case pausedByPressure = "paused_by_pressure"
        case autoPauseEnabled = "auto_pause_enabled"
        case autoCleanupEnabled = "auto_cleanup_enabled"
        case memory
        case resumed
    }

    /// Convenience: returns `actionsTaken` falling back to the `last_actions` key.
    /// The settings and resume endpoints use `last_actions`; the main endpoint uses `actions_taken`.
    var actions: [String] { actionsTaken ?? [] }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        pressureLevel = try container.decodeIfPresent(String.self, forKey: .pressureLevel)
        pausedByPressure = try container.decode(Bool.self, forKey: .pausedByPressure)
        autoPauseEnabled = try container.decode(Bool.self, forKey: .autoPauseEnabled)
        autoCleanupEnabled = try container.decode(Bool.self, forKey: .autoCleanupEnabled)
        memory = try container.decodeIfPresent(MemoryStats.self, forKey: .memory)
        resumed = try container.decodeIfPresent(Bool.self, forKey: .resumed)

        // Decode actions_taken, falling back to last_actions
        if let actions = try container.decodeIfPresent([String].self, forKey: .actionsTaken) {
            actionsTaken = actions
        } else {
            enum ExtraKeys: String, CodingKey {
                case lastActions = "last_actions"
            }
            let extra = try decoder.container(keyedBy: ExtraKeys.self)
            actionsTaken = try extra.decodeIfPresent([String].self, forKey: .lastActions)
        }
    }
}
