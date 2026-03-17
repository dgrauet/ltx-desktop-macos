import Foundation
import SwiftUI

@MainActor
class MemoryViewModel: ObservableObject {
    @Published var memoryStats: MemoryStats?
    @Published var systemInfo: SystemInfoResponse?
    @Published var pressureLevel: String = "normal"
    @Published var isQueuePausedByPressure: Bool = false
    @Published var autoPauseEnabled: Bool = true
    @Published var autoCleanupEnabled: Bool = true
    @Published var lastActions: [String] = []

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

                // Fetch pressure state alongside memory stats
                if let pressure = try? await service.memoryPressure() {
                    pressureLevel = pressure.pressureLevel ?? "normal"
                    isQueuePausedByPressure = pressure.pausedByPressure
                    autoPauseEnabled = pressure.autoPauseEnabled
                    autoCleanupEnabled = pressure.autoCleanupEnabled
                    lastActions = pressure.actions
                }

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

    // MARK: - Pressure Actions

    func toggleAutoPause(service: BackendService) {
        let newValue = !autoPauseEnabled
        Task {
            if let result = try? await service.updateMemoryPressureSettings(
                autoPauseEnabled: newValue, autoCleanupEnabled: nil
            ) {
                autoPauseEnabled = result.autoPauseEnabled
                isQueuePausedByPressure = result.pausedByPressure
            }
        }
    }

    func toggleAutoCleanup(service: BackendService) {
        let newValue = !autoCleanupEnabled
        Task {
            if let result = try? await service.updateMemoryPressureSettings(
                autoPauseEnabled: nil, autoCleanupEnabled: newValue
            ) {
                autoCleanupEnabled = result.autoCleanupEnabled
            }
        }
    }

    func resumeQueue(service: BackendService) {
        Task {
            if let result = try? await service.resumeMemoryPressurePause() {
                isQueuePausedByPressure = result.pausedByPressure
            }
        }
    }
}
