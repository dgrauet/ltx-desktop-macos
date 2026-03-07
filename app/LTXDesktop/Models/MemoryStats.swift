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
