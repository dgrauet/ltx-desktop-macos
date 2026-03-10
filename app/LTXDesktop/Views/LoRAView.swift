import SwiftUI
import UniformTypeIdentifiers

struct LoRAView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = LoRAViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Error banner
            if let errorMessage = vm.errorMessage {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(errorMessage)
                        .font(.callout)
                    Spacer()
                    Button {
                        vm.errorMessage = nil
                    } label: {
                        Image(systemName: "xmark")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.15))
            }

            // Info banner
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                    .font(.caption)
                Text("LoRAs must be compatible with LTX-2.3 latent space. Place .safetensors files in ~/.ltx-desktop/loras/")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.controlBackgroundColor).opacity(0.4))

            Divider()

            if vm.isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading LoRAs...")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if vm.loras.isEmpty {
                emptyState
            } else {
                loraList
            }
        }
        .navigationTitle("LoRA Models")
        .toolbar {
            ToolbarItem {
                Button {
                    vm.isImporting = true
                } label: {
                    Image(systemName: "plus")
                }
                .help("Import LoRA (.safetensors)")
            }
            ToolbarItem {
                Button {
                    Task { await vm.loadLoRAs(using: backendService) }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
        }
        .fileImporter(
            isPresented: $vm.isImporting,
            allowedContentTypes: [UTType(filenameExtension: "safetensors") ?? .data],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    Task { await vm.importLoRA(from: url, using: backendService) }
                }
            case .failure(let error):
                vm.errorMessage = "Import failed: \(error.localizedDescription)"
            }
        }
        .task {
            await vm.loadLoRAs(using: backendService)
        }
    }

    // MARK: - LoRA List

    private var loraList: some View {
        List(vm.loras) { lora in
            loraRow(lora)
        }
        .listStyle(.inset)
    }

    private func loraRow(_ lora: LoRAInfo) -> some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                // Icon in colored circle
                ZStack {
                    Circle()
                        .fill(Color.accentColor.opacity(0.15))
                        .frame(width: 36, height: 36)
                    Image(systemName: lora.typeIcon)
                        .font(.system(size: 16))
                        .foregroundStyle(Color.accentColor)
                }

                // Name + type
                VStack(alignment: .leading, spacing: 2) {
                    Text(lora.name)
                        .font(.body)
                        .fontWeight(.medium)
                    Text(lora.typeLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // Size badge
                Text(String(format: "%.0f MB", lora.sizeMb))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(.controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 4))

                // Incompatible warning
                if !lora.compatible {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                            .font(.caption)
                        Text("Incompatible")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }

                // Toggle
                Toggle("", isOn: Binding<Bool>(
                    get: { lora.loaded },
                    set: { _ in
                        Task { await vm.toggleLoRA(lora, using: backendService) }
                    }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                .disabled(!lora.compatible)
            }

            // Strength slider (shown when loaded)
            if lora.loaded {
                HStack(spacing: 8) {
                    Text("Strength")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 56, alignment: .leading)

                    Slider(
                        value: strengthBinding(for: lora),
                        in: 0.0...1.0,
                        step: 0.05
                    )

                    Text(String(format: "%.2f", lora.strength))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }
                .padding(.leading, 48)
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.vertical, 4)
        .animation(.easeInOut(duration: 0.2), value: lora.loaded)
    }

    /// Create a binding that updates strength locally and sends to backend on change.
    private func strengthBinding(for lora: LoRAInfo) -> Binding<Double> {
        Binding<Double>(
            get: { lora.strength },
            set: { newValue in
                // Update local state immediately
                if let idx = vm.loras.firstIndex(where: { $0.id == lora.id }) {
                    vm.loras[idx].strength = newValue
                }
                // Debounce: send to backend
                Task {
                    await vm.updateStrength(lora, strength: newValue, using: backendService)
                }
            }
        )
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "folder.badge.questionmark")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("No LoRAs found")
                .font(.title3)
                .fontWeight(.medium)
            Text("Add .safetensors files to ~/.ltx-desktop/loras/\nor use the + button to import")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button {
                vm.isImporting = true
            } label: {
                Label("Import LoRA", systemImage: "plus.circle")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
