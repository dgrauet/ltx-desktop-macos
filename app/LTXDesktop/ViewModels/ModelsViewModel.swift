import Foundation
import Combine

/// ViewModel for the Models management tab in Settings.
/// Handles listing, downloading, and deleting AI models.
@MainActor
class ModelsViewModel: ObservableObject {
    @Published var models: [ModelInfo] = []
    @Published var totalDiskGb: Double = 0.0
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    /// Tracks active downloads: model_id -> (download_id, progress, status)
    @Published var activeDownloads: [String: DownloadState] = [:]

    /// Model pending delete confirmation
    @Published var modelPendingDelete: ModelInfo?

    struct DownloadState {
        let downloadId: String
        var progress: Double
        var status: String
        var error: String?
    }

    private var pollTimers: [String: Task<Void, Never>] = [:]

    func loadModels(service: BackendService) {
        isLoading = true
        errorMessage = nil
        Task {
            do {
                let response = try await service.listModels()
                self.models = response.models
                self.totalDiskGb = response.totalDiskGb
            } catch {
                self.errorMessage = "Failed to load models: \(error.localizedDescription)"
            }
            self.isLoading = false
        }
    }

    func startDownload(modelId: String, service: BackendService) {
        guard activeDownloads[modelId] == nil else { return }
        errorMessage = nil
        Task {
            do {
                let response = try await service.downloadModel(modelId: modelId)
                self.activeDownloads[modelId] = DownloadState(
                    downloadId: response.downloadId,
                    progress: 0.0,
                    status: "pending"
                )
                self.startPollingProgress(
                    modelId: modelId,
                    downloadId: response.downloadId,
                    service: service
                )
            } catch {
                self.errorMessage = "Failed to start download: \(error.localizedDescription)"
            }
        }
    }

    func deleteModel(modelId: String, service: BackendService) {
        errorMessage = nil
        Task {
            do {
                _ = try await service.deleteModel(modelId: modelId)
                self.loadModels(service: service)
            } catch {
                self.errorMessage = "Failed to delete model: \(error.localizedDescription)"
            }
        }
    }

    func confirmDelete(model: ModelInfo) {
        modelPendingDelete = model
    }

    func cancelDelete() {
        modelPendingDelete = nil
    }

    func isDownloading(_ modelId: String) -> Bool {
        guard let state = activeDownloads[modelId] else { return false }
        return state.status == "pending" || state.status == "downloading"
    }

    func downloadProgress(_ modelId: String) -> Double {
        activeDownloads[modelId]?.progress ?? 0.0
    }

    // MARK: - Progress Polling

    private func startPollingProgress(
        modelId: String,
        downloadId: String,
        service: BackendService
    ) {
        pollTimers[modelId]?.cancel()
        pollTimers[modelId] = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_500_000_000) // 1.5s
                guard !Task.isCancelled else { break }

                do {
                    let status = try await service.downloadStatus(downloadId: downloadId)
                    self.activeDownloads[modelId] = DownloadState(
                        downloadId: downloadId,
                        progress: status.progress,
                        status: status.status,
                        error: status.error
                    )

                    if status.status == "completed" {
                        self.activeDownloads.removeValue(forKey: modelId)
                        self.loadModels(service: service)
                        break
                    } else if status.status == "failed" {
                        self.errorMessage = status.error ?? "Download failed"
                        self.activeDownloads.removeValue(forKey: modelId)
                        break
                    }
                } catch {
                    // Polling error, retry on next tick
                }
            }
        }
    }

    func stopAllPolling() {
        for (_, task) in pollTimers {
            task.cancel()
        }
        pollTimers.removeAll()
    }

    deinit {
        for (_, task) in pollTimers {
            task.cancel()
        }
    }
}
