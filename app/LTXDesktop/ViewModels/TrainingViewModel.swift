import Foundation
import SwiftUI

// MARK: - Supporting Types

struct ManifestRow: Identifiable {
    let id = UUID()
    var video: String
    var caption: String
}

// MARK: - TrainingViewModel

@MainActor
final class TrainingViewModel: ObservableObject {

    // MARK: - Dataset State

    @Published var datasets: [TrainingDataset] = []
    @Published var selectedDatasetId: String?
    @Published var manifestRows: [ManifestRow] = []
    @Published var manifestWarnings: [String] = []

    // MARK: - Run State

    @Published var runs: [TrainingRun] = []

    // MARK: - Live Training Progress

    @Published var liveStatus: String = ""
    @Published var liveStep: Int = 0
    @Published var liveTotal: Int = 0
    @Published var livePeakGb: Double = 0
    @Published var isTraining: Bool = false

    // MARK: - Preflight

    @Published var preflight: PreflightResult?

    // MARK: - Config Inputs

    @Published var lowRam: Bool = false {
        didSet { applyDefaults(lowRam: lowRam) }
    }
    @Published var rank: Int = 64
    @Published var steps: Int = 500

    // MARK: - Error

    @Published var errorMessage: String?

    // MARK: - Private

    /// The run/job IDs of the active training run, stored so cancel works.
    private var activeRunId: String?
    private var activeJobId: String?

    // MARK: - Datasets

    func loadDatasets(using service: BackendService) async {
        do {
            datasets = try await service.listDatasets()
        } catch {
            errorMessage = "Failed to load datasets: \(error.localizedDescription)"
        }
    }

    func createDataset(_ id: String, using service: BackendService) async {
        do {
            let dataset = try await service.createDataset(id: id)
            datasets.append(dataset)
            selectedDatasetId = dataset.id
        } catch {
            errorMessage = "Failed to create dataset: \(error.localizedDescription)"
        }
    }

    /// Upload each URL as a clip, then refresh manifestRows from the uploaded filenames.
    func addClips(_ urls: [URL], using service: BackendService) async {
        guard let datasetId = selectedDatasetId else {
            errorMessage = "Select a dataset before adding clips."
            return
        }
        for url in urls {
            do {
                try await service.uploadClip(datasetId: datasetId, fileURL: url)
                let filename = url.lastPathComponent
                // Only add if not already in the list.
                if !manifestRows.contains(where: { $0.video == filename }) {
                    manifestRows.append(ManifestRow(video: filename, caption: ""))
                }
            } catch {
                errorMessage = "Failed to upload \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }

    /// Write the current manifestRows to the backend and store any returned warnings.
    func saveManifest(using service: BackendService) async {
        guard let datasetId = selectedDatasetId else {
            errorMessage = "Select a dataset before saving the manifest."
            return
        }
        do {
            let entries = manifestRows.map { (video: $0.video, caption: $0.caption) }
            let result = try await service.putManifest(datasetId: datasetId, entries: entries)
            manifestWarnings = result.warnings
        } catch {
            errorMessage = "Failed to save manifest: \(error.localizedDescription)"
        }
    }

    func deleteDataset(_ id: String, using service: BackendService) async {
        do {
            try await service.deleteDataset(id: id)
            datasets.removeAll { $0.id == id }
            if selectedDatasetId == id {
                selectedDatasetId = nil
                manifestRows = []
                manifestWarnings = []
            }
        } catch {
            errorMessage = "Failed to delete dataset: \(error.localizedDescription)"
        }
    }

    // MARK: - Runs

    func loadRuns(using service: BackendService) async {
        do {
            runs = try await service.listRuns()
        } catch {
            errorMessage = "Failed to load runs: \(error.localizedDescription)"
        }
    }

    func deleteRun(_ run: TrainingRun, using service: BackendService) async {
        do {
            try await service.deleteRun(id: run.runId)
            runs.removeAll { $0.runId == run.runId }
        } catch {
            errorMessage = "Failed to delete run: \(error.localizedDescription)"
        }
    }

    // MARK: - Preflight

    func runPreflight(using service: BackendService) async {
        guard let datasetId = selectedDatasetId else {
            errorMessage = "Select a dataset before running preflight."
            return
        }
        let config = TrainingConfigRequest(
            datasetId: datasetId,
            lowRam: lowRam,
            rank: rank,
            steps: steps,
            learningRate: nil,
            seed: nil
        )
        do {
            preflight = try await service.preflight(config)
        } catch {
            errorMessage = "Preflight failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Training

    /// Start a training run. Stores `steps` as `liveTotal` because the backend
    /// never sends a non-zero `total` in step events — progress must be computed
    /// from the submitted config's step count.
    func startTraining(using service: BackendService) async {
        guard let datasetId = selectedDatasetId else {
            errorMessage = "Select a dataset before training."
            return
        }
        guard !isTraining else { return }

        let config = TrainingConfigRequest(
            datasetId: datasetId,
            lowRam: lowRam,
            rank: rank,
            steps: steps,
            learningRate: nil,
            seed: nil
        )

        // Store the submitted step count before we start, so progress can be computed
        // from it — the backend's step events carry total=0.
        liveTotal = steps
        liveStep = 0
        livePeakGb = 0
        liveStatus = "Starting…"
        isTraining = true
        errorMessage = nil

        do {
            let (runId, jobId) = try await service.startRun(config)
            activeRunId = runId
            activeJobId = jobId

            for await event in service.connectTraining(jobId: jobId, onEvent: { _ in }) {
                switch event {
                case .status(let message):
                    liveStatus = message

                case .step(let stepNum, _, let peakMem):
                    // Ignore the event's `total` — it is always 0 from the backend.
                    // Use liveTotal set from the submitted config's `steps`.
                    liveStep = stepNum
                    livePeakGb = peakMem
                    liveStatus = "Step \(stepNum) / \(liveTotal)"

                case .sample:
                    // Sample preview path — no action needed at VM level.
                    break

                case .done:
                    liveStatus = "Done"
                    isTraining = false
                    activeRunId = nil
                    activeJobId = nil
                    await loadRuns(using: service)

                case .error(let message):
                    errorMessage = message
                    liveStatus = "Error"
                    isTraining = false
                    activeRunId = nil
                    activeJobId = nil
                    await loadRuns(using: service)
                }
            }

            // Stream finished without a done/error event (e.g. connection closed).
            if isTraining {
                isTraining = false
                activeRunId = nil
                activeJobId = nil
                await loadRuns(using: service)
            }

        } catch {
            errorMessage = "Training failed: \(error.localizedDescription)"
            liveStatus = "Error"
            isTraining = false
            activeRunId = nil
            activeJobId = nil
        }
    }

    func cancelTraining(using service: BackendService) async {
        guard let runId = activeRunId else { return }
        do {
            try await service.cancelRun(id: runId)
            isTraining = false
            liveStatus = "Cancelled"
            activeRunId = nil
            activeJobId = nil
            await loadRuns(using: service)
        } catch {
            errorMessage = "Cancel failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Defaults

    /// Set rank/steps presets based on low-RAM mode.
    func applyDefaults(lowRam: Bool) {
        if lowRam {
            rank = 32
            steps = 300
        } else {
            rank = 64
            steps = 500
        }
    }
}
