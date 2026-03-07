import Foundation
import SwiftUI

@MainActor
class GenerationViewModel: ObservableObject {
    @Published var prompt = ""
    @Published var selectedResolution: Resolution = .landscape768
    @Published var numFrames = 97
    @Published var steps = 8
    @Published var seed = 42
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var errorMessage: String?
    @Published var outputVideoURL: URL?

    enum Resolution: String, CaseIterable, Identifiable {
        case landscape768 = "768×512"
        case portrait512 = "512×768"
        case landscape1280 = "1280×704"
        case portrait704 = "704×1280"
        case fullHD = "1920×1080"
        case portraitHD = "1080×1920"

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

        let request = T2VRequest(
            prompt: prompt,
            width: selectedResolution.width,
            height: selectedResolution.height,
            numFrames: numFrames,
            steps: steps,
            seed: seed
        )

        do {
            let jobResponse = try await service.generateTextToVideo(request: request)
            let jobId = jobResponse.jobId

            // Connect to WebSocket for progress
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

            // Get final job status
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
}
