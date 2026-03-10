import SwiftUI
import UniformTypeIdentifiers

@MainActor
class LoRAViewModel: ObservableObject {
    @Published var loras: [LoRAInfo] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var isImporting = false

    func loadLoRAs(using service: BackendService) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            loras = try await service.listLoRAs()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func toggleLoRA(_ lora: LoRAInfo, using service: BackendService) async {
        do {
            if lora.loaded {
                try await service.unloadLoRA(loraId: lora.id)
            } else {
                try await service.loadLoRA(loraId: lora.id, strength: lora.strength)
            }
            // Refresh list
            await loadLoRAs(using: service)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func updateStrength(_ lora: LoRAInfo, strength: Double, using service: BackendService) async {
        guard lora.loaded else { return }
        do {
            try await service.updateLoRAStrength(loraId: lora.id, strength: strength)
            // Update local state immediately for responsive UI
            if let idx = loras.firstIndex(where: { $0.id == lora.id }) {
                loras[idx].strength = strength
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func importLoRA(from url: URL, using service: BackendService) async {
        do {
            try await service.importLoRA(sourcePath: url.path)
            await loadLoRAs(using: service)
        } catch {
            errorMessage = "Import failed: \(error.localizedDescription)"
        }
    }
}
