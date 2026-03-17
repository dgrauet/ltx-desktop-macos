import Foundation

/// ViewModel for the dedicated Queue tab.
///
/// Polls GET /api/v1/queue every 2 seconds to keep the queue state fresh.
/// Provides cancel and priority-change actions for individual jobs.
@MainActor
class QueueViewModel: ObservableObject {
    @Published var entries: [QueueEntry] = []
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    private var pollTask: Task<Void, Never>?

    /// Number of jobs that are queued (not yet running).
    var queuedCount: Int {
        entries.filter { $0.state == "queued" }.count
    }

    /// Number of jobs currently running.
    var runningCount: Int {
        entries.filter { $0.state == "running" }.count
    }

    /// Total active jobs (running + queued) -- used for badge.
    var activeCount: Int {
        entries.filter { $0.state == "queued" || $0.state == "running" }.count
    }

    /// The currently running job, if any. Used for the persistent progress bar.
    var runningJob: QueueEntry? {
        entries.first { $0.state == "running" }
    }

    // MARK: - Polling

    func startPolling(service: BackendService) {
        stopPolling()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh(service: service)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    func refresh(service: BackendService) async {
        do {
            let queue = try await service.getQueue()
            if queue != entries { entries = queue }
            errorMessage = nil
        } catch {
            // Silently fail on polling — only set error on explicit refresh
        }
    }

    func explicitRefresh(service: BackendService) async {
        isLoading = true
        do {
            let queue = try await service.getQueue()
            if queue != entries { entries = queue }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    // MARK: - Actions

    func cancelJob(jobId: String, service: BackendService) async {
        do {
            let _ = try await service.cancelJob(jobId: jobId)
            await refresh(service: service)
        } catch {
            errorMessage = "Cancel failed: \(error.localizedDescription)"
        }
    }

    func changeJobPriority(jobId: String, priority: String, service: BackendService) async {
        do {
            let _ = try await service.changeJobPriority(jobId: jobId, priority: priority)
            await refresh(service: service)
        } catch {
            errorMessage = "Priority change failed: \(error.localizedDescription)"
        }
    }
}
