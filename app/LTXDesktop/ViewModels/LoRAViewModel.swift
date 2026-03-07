import SwiftUI

@MainActor
class LoRAViewModel: ObservableObject {
    @Published var loras: [LoRAInfo] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

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
                try await service.loadLoRA(loraId: lora.id)
            }
            // Refresh list
            await loadLoRAs(using: service)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
