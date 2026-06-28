import SwiftUI
import UniformTypeIdentifiers

struct LoRAView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = LoRAViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header with inline action buttons
            HStack {
                Text("LoRA Models")
                    .font(.title2)
                    .fontWeight(.semibold)

                Spacer()

                Button {
                    Task { await vm.loadLoRAs(using: backendService) }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh")

                Button {
                    vm.isImporting = true
                } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
                .help("Import LoRA (.safetensors)")
            }
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .padding(.bottom, 8)

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
                Text("Your LoRA library. Select and set strength per generation in the Generation panel. Place .safetensors files in ~/.ltx-desktop/loras/ (must be LTX-2.3 compatible).")
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
        ScrollView {
            VStack(spacing: 0) {
                ForEach(vm.loras) { lora in
                    loraRow(lora)
                    Divider()
                        .padding(.leading, 16)
                }
            }
            .padding(.vertical, 8)
        }
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
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
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
