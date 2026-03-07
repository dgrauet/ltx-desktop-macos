import Foundation
import SwiftUI

@MainActor
class MemoryViewModel: ObservableObject {
    @Published var memoryStats: MemoryStats?
    @Published var systemInfo: SystemInfoResponse?

    private var pollingTask: Task<Void, Never>?

    func startPolling(service: BackendService, isGenerating: Bool) {
        stopPolling()
        let interval: TimeInterval = isGenerating ? 2.0 : 10.0

        pollingTask = Task {
            // Fetch system info once
            if systemInfo == nil {
                systemInfo = try? await service.systemInfo()
            }

            while !Task.isCancelled {
                memoryStats = try? await service.memoryStats()
                try? await Task.sleep(for: .seconds(interval))
            }
        }
    }

    func stopPolling() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    // MARK: - Warning States

    var highCacheWarning: Bool {
        guard let stats = memoryStats else { return false }
        return stats.activeMemoryGb > 0 && stats.cacheMemoryGb > 2 * stats.activeMemoryGb
    }

    var criticalPeakWarning: Bool {
        guard let stats = memoryStats, let info = systemInfo else { return false }
        return stats.peakMemoryGb > Double(info.ramTotalGb) * 0.85
    }

    var lowAvailableWarning: Bool {
        guard let stats = memoryStats else { return false }
        return stats.systemAvailableGb < 4.0
    }
}
