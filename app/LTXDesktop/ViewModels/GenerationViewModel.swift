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
    @Published var seed = 42
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var errorMessage: String?
    @Published var outputVideoURL: URL?
    @Published var progressiveFrame: NSImage?
    @Published var sourceImagePath: String?
    @Published var sourceImageData: Data?
    @Published var isEnhancing: Bool = false
    @Published var imageStrength: Double = 0.85

    enum Resolution: String, CaseIterable, Identifiable {
        case landscape768 = "768x512"
        case portrait512 = "512x768"
        case landscape1280 = "1280x704"
        case portrait704 = "704x1280"
        case fullHD = "1920x1080"
        case portraitHD = "1080x1920"

        var id: String { rawValue }
        var label: String { rawValue }

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
    }

    func generate(using service: BackendService) async {
        guard !prompt.isEmpty else { return }

        isGenerating = true
        progress = 0
        errorMessage = nil
        outputVideoURL = nil
        progressiveFrame = nil

        do {
            let jobResponse: JobResponse

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
                    imageStrength: imageStrength
                )
                jobResponse = try await service.generateImageToVideo(request: request)
            } else {
                let request = T2VRequest(
                    prompt: prompt,
                    width: selectedResolution.width,
                    height: selectedResolution.height,
                    numFrames: numFrames,
                    steps: steps,
                    seed: seed,
                    fps: fps
                )
                jobResponse = try await service.generateTextToVideo(request: request)
            }

            let jobId = jobResponse.jobId

            for await update in service.connectProgress(jobId: jobId) {
                if let pct = update.pct {
                    progress = pct
                }
                if let frameB64 = update.previewFrame,
                   let frameData = Data(base64Encoded: frameB64) {
                    progressiveFrame = NSImage(data: frameData)
                }
                if let error = update.error {
                    errorMessage = error
                    isGenerating = false
                    return
                }
                if update.done == true {
                    break
                }
            }

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
    }

    func generatePreview(using service: BackendService) async {
        guard !prompt.isEmpty else { return }

        isGenerating = true
        progress = 0
        errorMessage = nil
        outputVideoURL = nil
        progressiveFrame = nil

        let request = PreviewRequest(
            prompt: prompt,
            seed: seed,
            sourceImagePath: sourceImagePath,
            imageStrength: imageStrength
        )

        do {
            let jobResponse = try await service.generatePreview(request: request)
            let jobId = jobResponse.jobId

            for await update in service.connectProgress(jobId: jobId) {
                if let pct = update.pct {
                    progress = pct
                }
                if let error = update.error {
                    errorMessage = error
                    isGenerating = false
                    return
                }
                if update.done == true {
                    break
                }
            }

            let status = try await service.getJobStatus(jobId: jobId)
            if status.status == "completed", let result = status.result {
                outputVideoURL = URL(fileURLWithPath: result.outputPath)
            } else if let error = status.error {
                errorMessage = error
            }

        } catch {
            errorMessage = error.localizedDescription
        }

        isGenerating = false
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
}
