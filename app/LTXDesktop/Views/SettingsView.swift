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
        TabView {
            generalTab
                .tabItem {
                    Label("General", systemImage: "slider.horizontal.3")
                }

            modelsTab
                .tabItem {
                    Label("Models", systemImage: "cube.box")
                }

            memoryTab
                .tabItem {
                    Label("Memory", systemImage: "memorychip")
                }

            loraTab
                .tabItem {
                    Label("LoRA", systemImage: "puzzlepiece.extension")
                }
        }
        .padding()
        .onAppear {
            memoryVM.startPolling(service: backendService, isGenerating: false)
            modelsVM.loadModels(service: backendService)
        }
        .onDisappear {
            memoryVM.stopPolling()
            modelsVM.stopAllPolling()
        }
    }

    // MARK: - General Tab

    private var generalTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text("General")
                    .font(.title2)
                    .fontWeight(.semibold)

                // Prompt Enhancement section
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
                .padding(14)
                .background(Color(.controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                // Output directory section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Output")
                        .font(.headline)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Output Directory")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)

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
                .padding(14)
                .background(Color(.controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                Spacer()
            }
            .padding()
        }
    }

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

    // MARK: - Models Tab

    private var modelsTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header with refresh button
                HStack {
                    Text("Models")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Spacer()

                    Button {
                        modelsVM.loadModels(service: backendService)
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(modelsVM.isLoading)
                }

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
                    .frame(maxWidth: .infinity, minHeight: 100)
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
                    .background(Color(.controlBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

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
                    .padding(.top, 4)
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

                Spacer()
            }
            .padding()
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
                    // Downloading state
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
                        Text("Downloaded")
                            .font(.caption)
                            .foregroundStyle(.green)

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

    // MARK: - Memory Tab

    private var memoryTab: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Metal Memory Monitor")
                .font(.title2)
                .fontWeight(.semibold)

            if let stats = memoryVM.memoryStats {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible()),
                ], spacing: 16) {
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

                Divider()

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
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Spacer()
        }
        .padding()
    }

    // MARK: - LoRA Tab

    private var loraTab: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("LoRA Models")
                    .font(.title2)
                    .fontWeight(.semibold)

                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) {
                        Image(systemName: "puzzlepiece.extension")
                            .foregroundStyle(.secondary)
                        Text("Manage LoRAs in the LoRA tab")
                            .font(.body)
                        Spacer()
                    }
                    Text("LoRAs extend the model with specialized capabilities: camera control, detail enhancement, and custom styles. Select the LoRA tab in the sidebar to browse, load, and toggle available LoRAs.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                            .font(.caption)
                        Text("LoRAs must be compatible with the LTX-2.3 latent space. LTX-2.0 LoRAs are not compatible.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)
                }
                .padding(14)
                .background(Color(.controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                Spacer()
            }
            .padding()
        }
    }

    // MARK: - Components

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
        .background(Color(.controlBackgroundColor).opacity(0.5))
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
