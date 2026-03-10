import SwiftUI
import AVKit
import UniformTypeIdentifiers

struct GenerationView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = GenerationViewModel()
    @AppStorage("promptEnhanceEnabled") private var enhanceEnabled: Bool = true
    @State private var player: AVPlayer?
    @State private var showQueuePopover = false

    var body: some View {
        HSplitView {
            // Left: Controls
            controlsPanel
                .frame(minWidth: 320, maxWidth: 400)

            // Right: Preview
            previewPanel
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding()
        .onChange(of: vm.outputVideoURL) { _, newURL in
            if let url = newURL {
                player = AVPlayer(url: url)
                player?.play()
            } else {
                player = nil
            }
        }
    }

    // MARK: - Controls Panel

    private var controlsPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Image drop zone for I2V
                imageDropZone

                // Prompt
                Text("Prompt")
                    .font(.headline)
                TextEditor(text: $vm.prompt)
                    .frame(minHeight: 100, maxHeight: 200)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .background(Color(.textBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )

                // I2V hint
                if vm.sourceImagePath != nil {
                    Text("Describe what's in the image and the motion you want. The prompt should match the image content.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 4)
                }

                // Word count + Enhance button
                HStack(alignment: .center) {
                    Spacer()
                    let wordCount = vm.prompt.split(separator: " ").count
                    Text("\(wordCount) words")
                        .font(.caption)
                        .foregroundStyle(
                            wordCount > 200 ? .red :
                            wordCount > 150 ? .orange :
                            Color.secondary
                        )

                    if enhanceEnabled {
                        Button(action: {
                            Task { await vm.enhancePrompt(using: backendService) }
                        }) {
                            HStack(spacing: 4) {
                                if vm.isEnhancing {
                                    ProgressView()
                                        .scaleEffect(0.6)
                                } else {
                                    Image(systemName: "sparkles")
                                }
                                Text(vm.isEnhancing ? "Enhancing..." : "Enhance")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(vm.prompt.isEmpty || vm.isEnhancing || vm.isGenerating)
                        .keyboardShortcut("e", modifiers: .command)
                    }
                }

                // Resolution
                HStack {
                    Text("Resolution")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.selectedResolution) {
                        ForEach(GenerationViewModel.Resolution.allCases) { res in
                            Text(res.label).tag(res)
                        }
                    }
                    .frame(width: 160)
                }

                // Frames
                HStack {
                    Text("Frames")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.numFrames) {
                        ForEach([9, 25, 49, 97], id: \.self) { n in
                            Text("\(n)").tag(n)
                        }
                    }
                    .frame(width: 80)
                }

                // FPS
                HStack {
                    Text("FPS")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.fps) {
                        Text("24").tag(24)
                        Text("30").tag(30)
                    }
                    .frame(width: 80)
                }

                // Steps
                HStack {
                    Text("Steps")
                        .font(.subheadline)
                    Spacer()
                    Stepper(value: $vm.steps, in: 1...50) {
                        Text("\(vm.steps)")
                            .monospacedDigit()
                    }
                }

                // Seed
                HStack {
                    Text("Seed")
                        .font(.subheadline)
                    Spacer()
                    TextField("Seed", value: $vm.seed, format: .number)
                        .frame(width: 100)
                        .textFieldStyle(.roundedBorder)
                }

                // Image Strength (only when I2V)
                if vm.sourceImagePath != nil {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Image Strength")
                                .font(.subheadline)
                            Spacer()
                            Text(String(format: "%.0f%%", vm.imageStrength * 100))
                                .font(.subheadline)
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $vm.imageStrength, in: 0.5...1.0, step: 0.05)
                        Text("Lower = more motion freedom, higher = closer to source image")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                // Neural Upscale toggle
                VStack(alignment: .leading, spacing: 4) {
                    Toggle(isOn: $vm.upscale) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Neural Upscale 2\u{00D7}")
                                .font(.subheadline)
                            Text("Generates at half resolution then upscales with neural network. Better quality at high resolutions.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    // Warning: high resolution without upscale
                    if !vm.upscale && vm.selectedResolution.needsUpscaleWarning {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .font(.caption)
                            Text("High resolution without upscale may be slow")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }

                    // Warning: Full HD on 32GB
                    if vm.selectedResolution.isFullHD {
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip")
                                .foregroundStyle(.red)
                                .font(.caption)
                            Text("Full HD requires 64GB+ RAM")
                                .font(.caption2)
                                .foregroundStyle(.red)
                        }
                    }
                }

                Divider()

                // Generate + Preview + Add to Queue buttons
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        Button(action: {
                            Task { await vm.generate(using: backendService) }
                        }) {
                            HStack {
                                if vm.isGenerating && vm.queuePosition == nil {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                }
                                Text(vm.isGenerating ? "Generating..." : "Generate")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(vm.prompt.isEmpty)
                        .keyboardShortcut("g", modifiers: .command)

                        Button(action: {
                            Task { await vm.generatePreview(using: backendService) }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "eye")
                                Text("Preview")
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(vm.prompt.isEmpty)
                        .keyboardShortcut("p", modifiers: .command)
                    }

                    // Add to Queue button — always enabled (that's the point of a queue)
                    Button(action: {
                        Task { await vm.addToQueue(using: backendService) }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "plus.circle")
                            Text("Add to Queue")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.regular)
                    .disabled(vm.prompt.isEmpty)
                    .keyboardShortcut("q", modifiers: [.command, .shift])
                }

                // Cancel button (when generating or queued)
                if vm.isGenerating, let jobId = vm.currentJobId {
                    Button(action: {
                        Task { await vm.cancelJob(jobId: jobId, using: backendService) }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "xmark.circle")
                            Text("Cancel")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.regular)
                    .keyboardShortcut(.escape, modifiers: [])
                }

                // Queue status indicator
                if vm.queueLength > 0 || !vm.queueEntries.isEmpty {
                    queueStatusBar
                }

                // Progress
                if vm.isGenerating {
                    VStack(alignment: .leading, spacing: 4) {
                        if let pos = vm.queuePosition, pos > 0 {
                            HStack(spacing: 4) {
                                Image(systemName: "clock")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                                Text("Position \(pos) in queue")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                        } else {
                            ProgressView(value: vm.progress)
                            HStack(spacing: 0) {
                                if let status = vm.statusMessage {
                                    Text("\(status) — ")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Text("\(Int(vm.progress * 100))%")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                // Error
                if let error = vm.errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .font(.caption)
                }

                Spacer()
            }
            .padding()
        }
    }

    // MARK: - Queue Status Bar

    private var queueStatusBar: some View {
        Button(action: { showQueuePopover.toggle() }) {
            HStack(spacing: 6) {
                Image(systemName: "list.number")
                    .font(.caption)
                let queuedCount = vm.queueEntries.filter { $0.state == "queued" }.count
                let runningCount = vm.queueEntries.filter { $0.state == "running" }.count
                if runningCount > 0 && queuedCount > 0 {
                    Text("1 running, \(queuedCount) queued")
                        .font(.caption)
                } else if runningCount > 0 {
                    Text("1 job running")
                        .font(.caption)
                } else if queuedCount > 0 {
                    Text("\(queuedCount) job\(queuedCount == 1 ? "" : "s") queued")
                        .font(.caption)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color(.controlBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showQueuePopover) {
            queuePopoverContent
        }
    }

    // MARK: - Queue Popover

    private var queuePopoverContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Generation Queue")
                    .font(.headline)
                Spacer()
                Button(action: {
                    Task { await vm.refreshQueue(service: backendService) }
                }) {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            if vm.queueEntries.isEmpty {
                Text("Queue is empty")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        ForEach(vm.queueEntries) { entry in
                            queueEntryRow(entry)
                            Divider()
                        }
                    }
                }
                .frame(maxHeight: 300)
            }
        }
        .frame(width: 340)
        .onAppear {
            Task { await vm.refreshQueue(service: backendService) }
        }
    }

    private func queueEntryRow(_ entry: QueueEntry) -> some View {
        HStack(spacing: 8) {
            // State indicator
            Circle()
                .fill(stateColor(entry.state))
                .frame(width: 8, height: 8)

            VStack(alignment: .leading, spacing: 2) {
                // Job type + prompt
                HStack(spacing: 4) {
                    Text(entry.jobType.uppercased())
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.accentColor.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 3))

                    Text(entry.prompt)
                        .font(.caption)
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                }

                // Priority + ETA
                HStack(spacing: 8) {
                    Text(entry.priority)
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    if entry.state == "running", let progress = entry.progress {
                        Text("\(Int(progress * 100))%")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else if entry.state == "queued" {
                        Text("Position \(entry.position)")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }

                    if let eta = entry.etaSeconds {
                        Text("~\(formatETA(eta))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Spacer()

            // Action buttons
            if entry.state == "queued" {
                // Priority buttons
                Menu {
                    Button("High Priority") {
                        Task { await vm.changeJobPriority(jobId: entry.jobId, priority: "high", using: backendService) }
                    }
                    Button("Normal Priority") {
                        Task { await vm.changeJobPriority(jobId: entry.jobId, priority: "normal", using: backendService) }
                    }
                    Button("Low Priority") {
                        Task { await vm.changeJobPriority(jobId: entry.jobId, priority: "low", using: backendService) }
                    }
                } label: {
                    Image(systemName: "arrow.up.arrow.down")
                        .font(.caption)
                }
                .menuStyle(.borderlessButton)
                .frame(width: 20)
            }

            // Cancel button
            if entry.state == "queued" || entry.state == "running" {
                Button(action: {
                    Task { await vm.cancelJob(jobId: entry.jobId, using: backendService) }
                }) {
                    Image(systemName: "xmark.circle")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func stateColor(_ state: String) -> Color {
        switch state {
        case "running": return .green
        case "queued": return .orange
        case "completed": return .blue
        case "failed": return .red
        case "cancelled": return .gray
        default: return .secondary
        }
    }

    private func formatETA(_ seconds: Double) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60))m"
        } else {
            return "\(Int(seconds / 3600))h"
        }
    }

    // MARK: - Image Drop Zone

    private var imageDropZone: some View {
        Group {
            if let imageData = vm.sourceImageData,
               let nsImage = NSImage(data: imageData) {
                // Show thumbnail + clear button
                HStack {
                    Image(nsImage: nsImage)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(height: 80)
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Source Image")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(vm.sourceImagePath?.components(separatedBy: "/").last ?? "image")
                            .font(.caption2)
                            .lineLimit(1)
                        Button("Clear") {
                            vm.clearSourceImage()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    Spacer()
                }
                .padding(8)
                .background(Color(.controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                // Drop zone
                VStack(spacing: 6) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                    Text("Drop image for Image-to-Video")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 70)
                .background(Color(.controlBackgroundColor).opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 3]))
                        .foregroundStyle(.secondary.opacity(0.4))
                )
                .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
                    handleDrop(providers: providers)
                    return true
                }
            }
        }
    }

    private func handleDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
            guard let urlData = data as? Data,
                  let url = URL(dataRepresentation: urlData, relativeTo: nil) else { return }

            let imageTypes = ["png", "jpg", "jpeg", "tiff", "heic", "webp"]
            guard imageTypes.contains(url.pathExtension.lowercased()) else { return }

            DispatchQueue.main.async {
                vm.handleImageDrop(urls: [url])
            }
        }
    }

    // MARK: - Preview Panel

    private var previewPanel: some View {
        Group {
            if let player = player {
                VideoPlayer(player: player)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else if vm.isGenerating, let frame = vm.progressiveFrame {
                // Progressive diffusion display during generation
                VStack(spacing: 8) {
                    Image(nsImage: frame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    Text("Generating...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "film")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                    Text("Generated video will appear here")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.windowBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }
}
