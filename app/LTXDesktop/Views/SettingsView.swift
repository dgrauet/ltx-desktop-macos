import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var memoryVM = MemoryViewModel()
    @StateObject private var modelsVM = ModelsViewModel()

    // General — Prompt Enhancement
    @AppStorage("promptEnhanceEnabled") private var enhanceEnabled: Bool = true

    // General — Output directory
    @AppStorage("outputDirectory") private var outputDirectory: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 32) {
                // Title
                Text("Settings")
                    .font(.title2)
                    .fontWeight(.semibold)

                // MARK: - General
                settingsSection(title: "General", icon: "slider.horizontal.3") {
                    // Prompt Enhancement
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Prompt Enhancement")
                            .font(.headline)

                        Toggle(isOn: $enhanceEnabled) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Enable prompt enhancement (Qwen3.5-2B)")
                                    .font(.body)
                                Text("Automatically expands short prompts into detailed LTX-2.3 optimized descriptions. Requires ~1.2GB RAM and temporarily unloads the video model.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                        }
                        .toggleStyle(.switch)
                    }

                    Divider()

                    // Output directory
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Output Directory")
                            .font(.headline)

                        HStack(spacing: 8) {
                            TextField(
                                "~/.ltx-desktop/outputs",
                                text: $outputDirectory
                            )
                            .textFieldStyle(.roundedBorder)

                            Button("Choose...") {
                                chooseOutputDirectory()
                            }
                            .buttonStyle(.bordered)
                        }

                        if outputDirectory.isEmpty {
                            Text("Default: ~/.ltx-desktop/outputs")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                // MARK: - Models
                settingsSection(title: "Models", icon: "cube.box") {
                    // Error banner
                    if let error = modelsVM.errorMessage {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.caption)
                            Spacer()
                            Button("Dismiss") {
                                modelsVM.errorMessage = nil
                            }
                            .font(.caption)
                            .buttonStyle(.borderless)
                        }
                        .padding(10)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    if modelsVM.isLoading && modelsVM.models.isEmpty {
                        VStack(spacing: 12) {
                            ProgressView()
                            Text("Loading model information...")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, minHeight: 80)
                    } else {
                        // Model list
                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(modelsVM.models.enumerated()), id: \.element.id) { index, model in
                                modelRow(model)

                                if index < modelsVM.models.count - 1 {
                                    Divider()
                                        .padding(.leading, 16)
                                }
                            }
                        }
                        .background(Color(.controlBackgroundColor).opacity(0.3))
                        .clipShape(RoundedRectangle(cornerRadius: 8))

                        // Disk usage summary
                        HStack(spacing: 16) {
                            HStack(spacing: 6) {
                                Image(systemName: "internaldrive")
                                    .foregroundStyle(.secondary)
                                    .font(.caption)
                                Text("Total disk usage: \(String(format: "%.1f GB", modelsVM.totalDiskGb))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()

                            let downloadedCount = modelsVM.models.filter(\.downloaded).count
                            Text("\(downloadedCount) of \(modelsVM.models.count) models downloaded")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    HStack(spacing: 6) {
                        Image(systemName: "info.circle")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                        Text("Models are stored in ~/.cache/huggingface/. Deleting a model frees disk space but requires re-download before next use.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                // MARK: - Memory
                settingsSection(title: "Metal Memory", icon: "memorychip") {
                    if let stats = memoryVM.memoryStats {
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible()),
                        ], spacing: 12) {
                            memoryCard(
                                title: "Active Memory",
                                value: stats.activeMemoryGb,
                                color: .blue
                            )
                            memoryCard(
                                title: "Cache Memory",
                                value: stats.cacheMemoryGb,
                                color: memoryVM.highCacheWarning ? .yellow : .green
                            )
                            memoryCard(
                                title: "Peak Memory",
                                value: stats.peakMemoryGb,
                                color: memoryVM.criticalPeakWarning ? .red : .orange
                            )
                            memoryCard(
                                title: "System Available",
                                value: stats.systemAvailableGb,
                                color: memoryVM.lowAvailableWarning ? .red : .mint
                            )
                        }

                        // Reload counter
                        HStack {
                            Text("Generations since reload:")
                                .foregroundStyle(.secondary)
                            Text("\(stats.generationCountSinceReload) / 5")
                                .fontWeight(.medium)
                                .monospacedDigit()
                            Spacer()
                            Text("Next reload in \(stats.nextReloadIn) generations")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        // Warnings
                        if memoryVM.highCacheWarning {
                            warningBanner(
                                message: "High cache — cleanup recommended",
                                color: .yellow
                            )
                        }
                        if memoryVM.criticalPeakWarning {
                            warningBanner(
                                message: "Memory critical — peak exceeds 85% of RAM",
                                color: .red
                            )
                        }
                        if memoryVM.lowAvailableWarning {
                            warningBanner(
                                message: "Low memory — reduce resolution or frame count",
                                color: .red
                            )
                        }
                    } else {
                        VStack(spacing: 12) {
                            ProgressView()
                            Text("Loading memory stats...")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, minHeight: 80)
                    }
                }

                // MARK: - LoRA Info
                settingsSection(title: "LoRA", icon: "puzzlepiece.extension") {
                    HStack(spacing: 8) {
                        Image(systemName: "puzzlepiece.extension")
                            .foregroundStyle(.secondary)
                        Text("Manage LoRAs in the LoRA tab in the sidebar.")
                            .font(.body)
                        Spacer()
                    }

                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                            .font(.caption)
                        Text("LoRAs must be compatible with the LTX-2.3 latent space. LTX-2.0 LoRAs are not compatible.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(20)
        }
        .onAppear {
            memoryVM.startPolling(service: backendService, isGenerating: false)
            modelsVM.loadModels(service: backendService)
        }
        .onDisappear {
            memoryVM.stopPolling()
            modelsVM.stopAllPolling()
        }
        .alert(
            "Delete Model",
            isPresented: Binding(
                get: { modelsVM.modelPendingDelete != nil },
                set: { if !$0 { modelsVM.cancelDelete() } }
            ),
            presenting: modelsVM.modelPendingDelete
        ) { model in
            Button("Cancel", role: .cancel) {
                modelsVM.cancelDelete()
            }
            Button("Delete", role: .destructive) {
                modelsVM.deleteModel(modelId: model.id, service: backendService)
                modelsVM.cancelDelete()
            }
        } message: { model in
            Text("Are you sure you want to delete \"\(model.name)\"? This will free approximately \(model.sizeLabel) of disk space. You will need to re-download the model before generating videos.")
        }
    }

    // MARK: - Section Builder

    private func settingsSection<Content: View>(
        title: String,
        icon: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: icon)
                .font(.headline)

            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Components

    private func chooseOutputDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose Output Directory"
        panel.message = "Select where generated videos will be saved."

        if panel.runModal() == .OK, let url = panel.url {
            outputDirectory = url.path
        }
    }

    private func modelRow(_ model: ModelInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                // Status indicator
                Circle()
                    .fill(model.downloaded ? Color.green : Color(.systemGray))
                    .frame(width: 10, height: 10)

                // Name + description
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(model.name)
                            .font(.body)
                            .fontWeight(.medium)

                        Text(model.typeLabel)
                            .font(.caption2)
                            .fontWeight(.medium)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }

                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                // Size badge
                Text(model.sizeLabel)
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.accentColor.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 5))

                // Action buttons
                if modelsVM.isDownloading(model.id) {
                    VStack(spacing: 2) {
                        ProgressView(value: modelsVM.downloadProgress(model.id))
                            .frame(width: 80)
                        Text("\(Int(modelsVM.downloadProgress(model.id) * 100))%")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                } else if model.downloaded {
                    HStack(spacing: 8) {
                        // "Use" button for video generators
                        if model.modelType == "video_generator" {
                            if modelsVM.isSelected(model) {
                                Text("Active")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundStyle(.white)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Color.accentColor)
                                    .clipShape(RoundedRectangle(cornerRadius: 5))
                            } else {
                                Button {
                                    modelsVM.selectModel(modelId: model.id, service: backendService)
                                } label: {
                                    Text("Use")
                                        .font(.caption)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        } else {
                            Text("Downloaded")
                                .font(.caption)
                                .foregroundStyle(.green)
                        }

                        Button {
                            modelsVM.confirmDelete(model: model)
                        } label: {
                            Image(systemName: "trash")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.red.opacity(0.7))
                        .help("Delete model to free disk space")
                    }
                } else {
                    Button {
                        modelsVM.startDownload(modelId: model.id, service: backendService)
                    } label: {
                        Label("Download", systemImage: "arrow.down.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private func memoryCard(title: String, value: Double, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(String(format: "%.2f GB", value))
                .font(.title3)
                .fontWeight(.semibold)
                .monospacedDigit()
            Gauge(value: min(value, 64), in: 0...64) {
                EmptyView()
            }
            .tint(color)
        }
        .padding(12)
        .background(Color(.controlBackgroundColor).opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func warningBanner(message: String, color: Color) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(color)
            Text(message)
                .font(.callout)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
