import Foundation
import SwiftUI
import AppKit

@MainActor
class GenerationViewModel: ObservableObject {
    @Published var prompt = ""
    @Published var selectedResolution: Resolution = .landscape768
    @Published var numFrames = 97
    @Published var steps = 8
    @Published var fps = 24
    @Published var seed = -1
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var errorMessage: String?
    @Published var outputVideoURL: URL?
    @Published var progressiveFrame: NSImage?
    @Published var sourceImagePath: String?
    @Published var sourceImageData: Data?
    @Published var isEnhancing: Bool = false
    @Published var imageStrength: Double = 1.0
    @Published var upscale: Bool = false
    @Published var statusMessage: String?

    // Queue state
    @Published var queueLength: Int = 0
    @Published var queuePosition: Int? = nil
    @Published var queueEntries: [QueueEntry] = []
    @Published var currentJobId: String? = nil

    // Hardware limits
    @Published var hardwareLimits: HardwareLimitsResponse?

    // LoRA selection
    @Published var availableLoRAs: [LoRAInfo] = []
    @Published var selectedLoRAIds: Set<String> = []

    // Presets
    @Published var presets: [GenerationPreset] = []
    @Published var selectedPresetId: String? = nil

    private var queuePollTask: Task<Void, Never>? = nil

    enum Resolution: String, CaseIterable, Identifiable {
        case landscape768 = "768x512"
        case portrait512 = "512x768"
        case landscape1280 = "1280x704"
        case portrait704 = "704x1280"
        case fullHD = "1920x1080"
        case portraitHD = "1080x1920"

        var id: String { rawValue }

        var label: String {
            switch self {
            case .landscape768: return "768\u{00D7}512"
            case .portrait512: return "512\u{00D7}768"
            case .landscape1280: return "1280\u{00D7}704 HD"
            case .portrait704: return "704\u{00D7}1280 HD"
            case .fullHD: return "1920\u{00D7}1080 Full HD"
            case .portraitHD: return "1080\u{00D7}1920 Full HD"
            }
        }

        var width: Int {
            switch self {
            case .landscape768: return 768
            case .portrait512: return 512
            case .landscape1280: return 1280
            case .portrait704: return 704
            case .fullHD: return 1920
            case .portraitHD: return 1080
            }
        }

        var height: Int {
            switch self {
            case .landscape768: return 512
            case .portrait512: return 768
            case .landscape1280: return 704
            case .portrait704: return 1280
            case .fullHD: return 1080
            case .portraitHD: return 1920
            }
        }

        /// Whether this resolution benefits from the two-stage upscale pipeline.
        var needsUpscaleWarning: Bool {
            switch self {
            case .landscape1280, .portrait704, .fullHD, .portraitHD: return true
            default: return false
            }
        }

        /// Whether this is a Full HD resolution (needs 64GB+ RAM warning).
        var isFullHD: Bool {
            switch self {
            case .fullHD, .portraitHD: return true
            default: return false
            }
        }
    }

    // MARK: - Generate (normal priority)

    func generate(using service: BackendService) async {
        await submitGeneration(using: service, priority: "normal")
    }

    // MARK: - Add to Queue (low priority)

    func addToQueue(using service: BackendService) async {
        await submitGeneration(using: service, priority: "low")
    }

    // MARK: - Submit Generation (shared logic)

    private func submitGeneration(using service: BackendService, priority: String) async {
        guard !prompt.isEmpty else { return }

        // Only take over the UI for normal/high priority when no job running
        let isActiveJob = priority != "low" || !isGenerating

        if isActiveJob {
            isGenerating = true
            progress = 0
            errorMessage = nil
            outputVideoURL = nil
            progressiveFrame = nil
            statusMessage = nil
        }

        do {
            let submitResponse: QueueSubmitResponse

            if let imagePath = sourceImagePath {
                let request = I2VRequest(
                    prompt: prompt,
                    sourceImagePath: imagePath,
                    width: selectedResolution.width,
                    height: selectedResolution.height,
                    numFrames: numFrames,
                    steps: steps,
                    seed: seed,
                    fps: fps,
                    imageStrength: imageStrength,
                    upscale: upscale,
                    loraIds: selectedLoRAIdArray
                )
                submitResponse = try await service.generateImageToVideo(request: request, priority: priority)
            } else {
                let request = T2VRequest(
                    prompt: prompt,
                    width: selectedResolution.width,
                    height: selectedResolution.height,
                    numFrames: numFrames,
                    steps: steps,
                    seed: seed,
                    fps: fps,
                    upscale: upscale,
                    loraIds: selectedLoRAIdArray
                )
                submitResponse = try await service.generateTextToVideo(request: request, priority: priority)
            }

            let jobId = submitResponse.jobId
            queueLength = submitResponse.queueLength
            queuePosition = submitResponse.position > 0 ? submitResponse.position : nil

            if isActiveJob {
                currentJobId = jobId
                startQueuePolling(service: service)
                await trackJob(jobId: jobId, service: service)
            } else {
                // Low priority batch — just refresh queue state
                await refreshQueue(service: service)
            }

        } catch {
            if isActiveJob {
                errorMessage = error.localizedDescription
                isGenerating = false
                statusMessage = nil
            }
        }
    }

    // MARK: - Track Active Job via WebSocket

    private func trackJob(jobId: String, service: BackendService) async {
        for await update in service.connectProgress(jobId: jobId) {
            if let pct = update.pct {
                progress = pct
            }
            if let newStatus = update.status {
                statusMessage = newStatus
            }
            if let ql = update.queueLength {
                queueLength = ql
            }
            if let pos = update.position {
                queuePosition = pos > 0 ? pos : nil
            }
            if let frameB64 = update.previewFrame,
               let frameData = Data(base64Encoded: frameB64) {
                progressiveFrame = NSImage(data: frameData)
            }
            if let error = update.error {
                errorMessage = error
                isGenerating = false
                stopQueuePolling()
                return
            }
            if update.done == true {
                break
            }
        }

        // Job finished — get result
        do {
            let status = try await service.getJobStatus(jobId: jobId)
            if status.status == "completed", let result = status.result {
                outputVideoURL = URL(fileURLWithPath: result.outputPath)
                progressiveFrame = nil
            } else if let error = status.error {
                errorMessage = error
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
        statusMessage = nil
        queuePosition = nil
        currentJobId = nil
        stopQueuePolling()
    }

    func generatePreview(using service: BackendService) async {
        guard !prompt.isEmpty else { return }

        isGenerating = true
        progress = 0
        errorMessage = nil
        outputVideoURL = nil
        progressiveFrame = nil
        statusMessage = nil

        let request = PreviewRequest(
            prompt: prompt,
            seed: seed,
            sourceImagePath: sourceImagePath,
            imageStrength: imageStrength,
            upscale: upscale,
            loraIds: selectedLoRAIdArray
        )

        do {
            let submitResponse = try await service.generatePreview(request: request)
            let jobId = submitResponse.jobId
            currentJobId = jobId
            queueLength = submitResponse.queueLength

            await trackJob(jobId: jobId, service: service)

        } catch {
            errorMessage = error.localizedDescription
            isGenerating = false
            statusMessage = nil
        }
    }

    func enhancePrompt(using service: BackendService) async {
        guard !prompt.isEmpty else { return }
        defer { isEnhancing = false }
        isEnhancing = true
        errorMessage = nil
        do {
            let response = try await service.enhancePrompt(prompt: prompt, isI2V: sourceImagePath != nil)
            prompt = response.enhanced
        } catch {
            errorMessage = "Enhance failed: \(error.localizedDescription)"
        }
    }

    func handleImageDrop(urls: [URL]) {
        guard let url = urls.first else { return }
        sourceImagePath = url.path
        sourceImageData = try? Data(contentsOf: url)
    }

    func clearSourceImage() {
        sourceImagePath = nil
        sourceImageData = nil
    }

    // MARK: - Queue Management

    func cancelJob(jobId: String, using service: BackendService) async {
        do {
            let _ = try await service.cancelJob(jobId: jobId)
            if jobId == currentJobId {
                isGenerating = false
                statusMessage = nil
                currentJobId = nil
                errorMessage = "Generation cancelled"
            }
            await refreshQueue(service: service)
        } catch {
            errorMessage = "Cancel failed: \(error.localizedDescription)"
        }
    }

    func cancelCurrentJob(using service: BackendService) async {
        guard let jobId = currentJobId else { return }
        await cancelJob(jobId: jobId, using: service)
    }

    func changeJobPriority(jobId: String, priority: String, using service: BackendService) async {
        do {
            let _ = try await service.changeJobPriority(jobId: jobId, priority: priority)
            await refreshQueue(service: service)
        } catch {
            errorMessage = "Priority change failed: \(error.localizedDescription)"
        }
    }

    func refreshQueue(service: BackendService) async {
        do {
            queueEntries = try await service.getQueue()
            queueLength = queueEntries.filter { $0.state == "queued" }.count
        } catch {
            // Silently fail — queue polling is best-effort
        }
    }

    // MARK: - Queue Polling

    private func startQueuePolling(service: BackendService) {
        stopQueuePolling()
        queuePollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshQueue(service: service)
                try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
            }
        }
    }

    private func stopQueuePolling() {
        queuePollTask?.cancel()
        queuePollTask = nil
    }

    // MARK: - Hardware Limits

    func fetchHardwareLimits(service: BackendService) async {
        hardwareLimits = try? await service.hardwareLimits()
    }

    // MARK: - LoRA Selection

    /// Fetch available LoRAs from the backend. Only shows loaded (active) LoRAs as selected.
    func fetchLoRAs(service: BackendService) async {
        do {
            let loras = try await service.listLoRAs()
            availableLoRAs = loras.filter { $0.compatible }
            // Auto-select LoRAs that are currently loaded in the backend
            selectedLoRAIds = Set(loras.filter { $0.loaded }.map { $0.id })
        } catch {
            // Silently fail — LoRA fetching is best-effort
        }
    }

    /// Toggle a LoRA's selection for the next generation.
    func toggleLoRASelection(_ loraId: String) {
        if selectedLoRAIds.contains(loraId) {
            selectedLoRAIds.remove(loraId)
        } else {
            selectedLoRAIds.insert(loraId)
        }
    }

    /// Array of selected LoRA IDs for inclusion in generation requests.
    var selectedLoRAIdArray: [String] {
        Array(selectedLoRAIds)
    }

    /// Whether RAM is below 32GB (show red warning banner).
    var isLowRAM: Bool {
        guard let limits = hardwareLimits else { return false }
        return limits.totalRamGb < 32
    }

    /// Warning message for the current resolution/frame combination, or nil if within limits.
    var resolutionWarning: String? {
        guard let limits = hardwareLimits else { return nil }
        let w = selectedResolution.width
        let h = selectedResolution.height

        // Check if this resolution exists in the limits
        if let limit = limits.resolutionLimits.first(where: { $0.width == w && $0.height == h }) {
            if numFrames > limit.maxFrames {
                return "Too many frames for your hardware (\(limits.totalRamGb)GB RAM). Recommended max: \(limit.maxFrames) frames at this resolution."
            }
            return nil
        }

        // Resolution not in limits at all — exceeds hardware
        return "This resolution may cause out-of-memory on your hardware (\(limits.totalRamGb)GB RAM)."
    }

    /// Maximum recommended frames for the current resolution.
    var recommendedMaxFrames: Int? {
        guard let limits = hardwareLimits else { return nil }
        let w = selectedResolution.width
        let h = selectedResolution.height
        return limits.resolutionLimits.first(where: { $0.width == w && $0.height == h })?.maxFrames
    }

    // MARK: - Presets

    func loadPresets(service: BackendService) async {
        do {
            presets = try await service.listPresets()
        } catch {
            // Silently fail — presets are non-critical
        }
    }

    func saveCurrentAsPreset(name: String, using service: BackendService) async {
        let params = PresetParams(
            width: selectedResolution.width,
            height: selectedResolution.height,
            numFrames: numFrames,
            steps: steps,
            seed: seed,
            fps: fps,
            guidanceScale: 1.0,
            negativePrompt: "",
            generateAudio: false,
            ffmpegUpscale: upscale
        )
        do {
            let preset = try await service.createPreset(name: name, params: params)
            presets.append(preset)
            selectedPresetId = preset.id
        } catch {
            errorMessage = "Failed to save preset: \(error.localizedDescription)"
        }
    }

    func applyPreset(_ preset: GenerationPreset) {
        let p = preset.params
        // Find matching resolution enum case
        if let res = Resolution.allCases.first(where: { $0.width == p.width && $0.height == p.height }) {
            selectedResolution = res
        }
        numFrames = p.numFrames
        steps = p.steps
        seed = p.seed
        fps = p.fps
        upscale = p.ffmpegUpscale ?? false
        selectedPresetId = preset.id
    }

    func deletePreset(_ presetId: String, using service: BackendService) async {
        do {
            try await service.deletePreset(presetId: presetId)
            presets.removeAll { $0.id == presetId }
            if selectedPresetId == presetId {
                selectedPresetId = nil
            }
        } catch {
            errorMessage = "Failed to delete preset: \(error.localizedDescription)"
        }
    }
}
