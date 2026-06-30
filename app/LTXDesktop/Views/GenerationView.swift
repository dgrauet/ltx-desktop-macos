import SwiftUI
import AVKit
import UniformTypeIdentifiers

/// AppKit-backed video player.
///
/// Replaces SwiftUI's `VideoPlayer`: its `_AVKit_SwiftUI` generic metadata
/// instantiation crashed the app (SIGABRT in `getSuperclassMetadata`) on
/// macOS 26.5.1 when the player first appeared after a generation.
/// Wrapping `AVPlayerView` directly skips that machinery entirely.
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player {
            view.player = player
        }
    }
}

struct GenerationView: View {
    @EnvironmentObject var backendService: BackendService
    @EnvironmentObject var vm: GenerationViewModel
    @AppStorage("promptEnhanceEnabled") private var enhanceEnabled: Bool = true
    @State private var player: AVPlayer?
    @State private var showQueuePopover = false
    @State private var showRetakeSheet = false
    @State private var showExtendSheet = false
    @State private var showSavePresetAlert = false
    @State private var newPresetName = ""
    @State private var showControlLibrary = false
    @StateObject private var controlLibraryVM = ControlLibraryViewModel()

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
        .task {
            await vm.fetchHardwareLimits(service: backendService)
            await vm.loadPresets(service: backendService)
            await vm.fetchLoRAs(service: backendService)
            await vm.loadICLoras(service: backendService)
        }
        .alert("Save Preset", isPresented: $showSavePresetAlert) {
            TextField("Preset name", text: $newPresetName)
            Button("Save") {
                let name = newPresetName.trimmingCharacters(in: .whitespaces)
                guard !name.isEmpty else { return }
                Task {
                    await vm.saveCurrentAsPreset(name: name, using: backendService)
                    newPresetName = ""
                }
            }
            Button("Cancel", role: .cancel) {
                newPresetName = ""
            }
        }
    }

    // MARK: - Controls Panel

    private var controlsPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Low RAM warning banner
                if vm.isLowRAM {
                    lowRAMBanner
                }

                // Preset bar
                presetBar

                // Image (I2V) + Audio (A2V, beta) drop zones side by side
                HStack(alignment: .top, spacing: 12) {
                    imageDropZone
                    audioDropZone
                }

                // Control-video drop zone for IC-LoRA
                controlVideoDropZone

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
                    // Word count + Enhance floated in the bottom-right of the field
                    .overlay(alignment: .bottomTrailing) {
                        HStack(spacing: 6) {
                            let wordCount = vm.prompt.split(separator: " ").count
                            Text("\(wordCount) words")
                                .font(.caption2)
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
                        .padding(.horizontal, 6)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(6)
                    }

                // I2V hint
                if vm.sourceImagePath != nil {
                    Text("Describe what's in the image and the motion you want. The prompt should match the image content.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 4)
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

                // Frames + FPS
                HStack {
                    Text("Frames")
                        .font(.subheadline)
                    Picker("", selection: $vm.numFrames) {
                        ForEach([9, 25, 49, 97, 161, 257], id: \.self) { n in
                            let secs = String(format: "%.1fs", Double(n) / Double(vm.fps))
                            Text("\(n) (\(secs))").tag(n)
                        }
                    }
                    .frame(width: 130)

                    Spacer()

                    Text("FPS")
                        .font(.subheadline)
                    Picker("", selection: $vm.fps) {
                        Text("24").tag(24)
                        Text("30").tag(30)
                    }
                    .frame(width: 70)
                }

                // Hardware warning for resolution/frames
                if let warning = vm.resolutionWarning {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                        Text(warning)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.horizontal, 4)
                }

                // Pipeline type — hidden in A2V mode (A2V uses its own two-stage pipeline)
                if vm.sourceAudioPath == nil && vm.controlVideoPath == nil {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Pipeline")
                                .font(.subheadline)
                            Picker("", selection: $vm.pipelineType) {
                                Text("Distilled (fast)").tag("distilled")
                                Text("One Stage (dev)").tag("one-stage")
                                Text("Two Stage (dev + upscale)").tag("two-stage")
                                Text("Two Stage HQ").tag("two-stage-hq")
                            }
                            .pickerStyle(.menu)
                        }
                        Text(pipelineDescription)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("Audio-to-Video uses a dedicated two-stage pipeline (beta). Pipeline selection is disabled.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Low RAM mode (DiT block streaming)
                VStack(alignment: .leading, spacing: 4) {
                    Toggle("Low RAM mode", isOn: $vm.lowRam)
                        .toggleStyle(.switch)
                        .controlSize(.small)
                    Text("Streams model weights from disk (~75% less RAM). Slower, but enables larger models on 16–32 GB Macs.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Steps + Seed
                HStack {
                    Text("Steps")
                        .font(.subheadline)
                    Stepper(value: $vm.steps, in: 1...50) {
                        Text("\(vm.steps)")
                            .monospacedDigit()
                    }

                    Spacer()

                    Text("Seed")
                        .font(.subheadline)
                    TextField("Seed", value: $vm.seed, format: .number)
                        .frame(width: 80)
                        .textFieldStyle(.roundedBorder)
                }

                // Guidance scale (CFG) — dev/two-stage pipelines and A2V use CFG;
                // the distilled pipeline ignores it.
                if vm.pipelineType != "distilled" || vm.sourceAudioPath != nil || vm.controlVideoPath != nil {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Guidance")
                                .font(.subheadline)
                            Spacer()
                            Text(String(format: "%.1f", vm.guidanceScale))
                                .font(.subheadline)
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $vm.guidanceScale, in: 1.0...10.0, step: 0.5)
                        Text("CFG strength. Higher = follows the prompt more closely (3.0 is the default).")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                // IC-LoRA selection + strengths (only in IC-LoRA mode)
                if vm.controlVideoPath != nil {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Control type")
                                .font(.subheadline)
                            Spacer()
                            Picker("", selection: $vm.controlType) {
                                ForEach(ControlType.allCases) { t in
                                    Text(t.label).tag(t)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(maxWidth: 220)
                        }
                        Text(controlTypeHelp)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        HStack {
                            Text("IC-LoRA")
                                .font(.subheadline)
                            Spacer()
                            Picker("", selection: $vm.selectedICLoraId) {
                                Text("Union Control (default)").tag(String?.none)
                                ForEach(vm.availableICLoras) { lora in
                                    Text(lora.name).tag(Optional(lora.id))
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(maxWidth: 220)
                        }
                        if vm.availableICLoras.isEmpty {
                            Text("No extra IC-LoRAs downloaded. Get them in Settings → Models.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        sliderRow("Control strength", value: $vm.controlStrength, range: 0.0...1.0)
                        sliderRow("IC-LoRA strength", value: $vm.icLoraStrength, range: 0.0...2.0)
                        sliderRow("Conditioning", value: $vm.conditioningStrength, range: 0.0...1.0)
                    }
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

                // Full HD RAM warning
                if vm.selectedResolution.isFullHD {
                    if let limits = vm.hardwareLimits, limits.totalRamGb < 64 {
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip")
                                .foregroundStyle(.red)
                                .font(.caption)
                            Text("Full HD requires 64GB+ RAM (you have \(limits.totalRamGb)GB)")
                                .font(.caption2)
                                .foregroundStyle(.red)
                        }
                    } else if vm.hardwareLimits == nil {
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

                // LoRA selection
                loraSection

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
                            ProgressView(value: min(max(vm.progress, 0), 1))
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

    // MARK: - Preset Bar

    private var presetBar: some View {
        HStack(spacing: 8) {
            Picker("Preset", selection: Binding(
                get: { vm.selectedPresetId ?? "" },
                set: { newValue in
                    if newValue.isEmpty {
                        vm.selectedPresetId = nil
                    } else if let preset = vm.presets.first(where: { $0.id == newValue }) {
                        vm.applyPreset(preset)
                    }
                }
            )) {
                Text("Custom").tag("")
                ForEach(vm.presets) { preset in
                    HStack {
                        Text(preset.name)
                        if preset.builtin {
                            Text("(built-in)")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .tag(preset.id)
                }
            }
            .labelsHidden()

            Button(action: {
                showSavePresetAlert = true
            }) {
                Image(systemName: "star")
                    .font(.system(size: 13))
            }
            .buttonStyle(.borderless)
            .help("Save current settings as preset")

            if let selectedId = vm.selectedPresetId,
               let preset = vm.presets.first(where: { $0.id == selectedId }),
               !preset.builtin {
                Button(action: {
                    Task { await vm.deletePreset(selectedId, using: backendService) }
                }) {
                    Image(systemName: "trash")
                        .font(.system(size: 12))
                        .foregroundStyle(.red)
                }
                .buttonStyle(.borderless)
                .help("Delete this preset")
            }
        }
    }

    // MARK: - Low RAM Banner

    private var lowRAMBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.white)
                .font(.subheadline)
            VStack(alignment: .leading, spacing: 2) {
                Text("Insufficient RAM")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.white)
                if let limits = vm.hardwareLimits {
                    Text("\(limits.totalRamGb)GB detected — 32GB minimum recommended. Very limited capability.")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.9))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
        }
        .padding(10)
        .background(Color.red.opacity(0.85))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - LoRA Section

    @State private var loraExpanded: Bool = false

    private var loraSection: some View {
        DisclosureGroup(isExpanded: $loraExpanded) {
            if vm.availableLoRAs.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Text("Manage LoRAs in Settings")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            } else {
                VStack(spacing: 8) {
                    ForEach(vm.availableLoRAs) { lora in
                        let isSelected = vm.selectedLoRAIds.contains(lora.id)
                        VStack(alignment: .leading, spacing: 4) {
                            Toggle(isOn: Binding<Bool>(
                                get: { isSelected },
                                set: { _ in vm.toggleLoRASelection(lora.id) }
                            )) {
                                Text(lora.name)
                                    .font(.caption)
                                    .lineLimit(1)
                            }
                            .toggleStyle(.checkbox)

                            // Strength is set here — the Generation panel is the
                            // single place to activate and tune a LoRA.
                            if isSelected {
                                HStack(spacing: 6) {
                                    Text("Strength")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                    Slider(
                                        value: Binding<Double>(
                                            get: { lora.strength },
                                            set: { newValue in
                                                Task {
                                                    await vm.updateLoRAStrength(
                                                        lora.id,
                                                        strength: newValue,
                                                        service: backendService
                                                    )
                                                }
                                            }
                                        ),
                                        in: 0.0...1.0,
                                        step: 0.05
                                    )
                                    Text(String(format: "%.2f", lora.strength))
                                        .font(.caption2.monospacedDigit())
                                        .foregroundStyle(.secondary)
                                        .frame(width: 30, alignment: .trailing)
                                }
                                .padding(.leading, 18)
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        } label: {
            HStack(spacing: 4) {
                Text("LoRA")
                    .font(.subheadline)
                if !vm.selectedLoRAIds.isEmpty {
                    Text("\(vm.selectedLoRAIds.count)")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.accentColor)
                        .clipShape(Capsule())
                }
            }
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
                // Drop zone + click to browse
                VStack(spacing: 6) {
                    Image(systemName: "photo.badge.plus")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                    Text("Drop or click to add image")
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
                .onTapGesture {
                    showImagePicker()
                }
                .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
                    handleDrop(providers: providers)
                    return true
                }
            }
        }
    }

    private func showImagePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select an image for Image-to-Video"
        if panel.runModal() == .OK, let url = panel.url {
            vm.handleImageDrop(urls: [url])
        }
    }

    // MARK: - Audio Drop Zone (A2V, beta)

    private var audioDropZone: some View {
        Group {
            if let audioPath = vm.sourceAudioPath {
                HStack(spacing: 8) {
                    Image(systemName: "waveform")
                        .font(.title2)
                        .foregroundStyle(.purple)
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Text("Source Audio")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("BETA")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.purple.opacity(0.15))
                                .foregroundStyle(.purple)
                                .clipShape(RoundedRectangle(cornerRadius: 3))
                        }
                        Text(audioPath.components(separatedBy: "/").last ?? "audio")
                            .font(.caption2)
                            .lineLimit(1)
                        Button("Clear") {
                            vm.clearSourceAudio()
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
                VStack(spacing: 6) {
                    Image(systemName: "waveform.badge.plus")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                    Text("Drop or click to add audio (A2V, beta)")
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
                .onTapGesture {
                    showAudioPicker()
                }
                .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
                    handleAudioDrop(providers: providers)
                    return true
                }
            }
        }
    }

    private func showAudioPicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select an audio track for Audio-to-Video"
        if panel.runModal() == .OK, let url = panel.url {
            vm.handleAudioDrop(urls: [url])
        }
    }

    private func handleAudioDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
            guard let urlData = data as? Data,
                  let url = URL(dataRepresentation: urlData, relativeTo: nil) else { return }

            let audioTypes = ["wav", "mp3", "m4a", "aac", "flac", "aiff", "aif", "ogg"]
            guard audioTypes.contains(url.pathExtension.lowercased()) else { return }

            DispatchQueue.main.async {
                vm.handleAudioDrop(urls: [url])
            }
        }
    }

    // MARK: - Control Video Drop Zone (IC-LoRA)

    private var controlVideoDropZone: some View {
        Group {
            if let controlPath = vm.controlVideoPath {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: "square.stack.3d.down.right")
                            .font(.title2)
                            .foregroundStyle(.indigo)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Control Video (IC-LoRA)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(controlPath.components(separatedBy: "/").last ?? "video")
                                .font(.caption2)
                                .lineLimit(1)
                            HStack(spacing: 6) {
                                Button("Clear") { vm.clearControlVideo() }
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                                Button {
                                    showControlLibrary = true
                                } label: {
                                    Label("Library", systemImage: "square.stack.3d.up")
                                }
                                .buttonStyle(.bordered).controlSize(.small)
                            }
                        }
                        Spacer()
                    }
                }
                .padding(8)
                .background(Color(.controlBackgroundColor).opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "square.stack.3d.down.right")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                    Text("Drop or click to add control video (IC-LoRA)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button {
                        showControlLibrary = true
                    } label: {
                        Label("Library", systemImage: "square.stack.3d.up")
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 90)
                .background(Color(.controlBackgroundColor).opacity(0.3))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 3]))
                        .foregroundStyle(.secondary.opacity(0.4))
                )
                .onTapGesture { showControlVideoPicker() }
                .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in
                    handleControlVideoDrop(providers: providers)
                    return true
                }
            }
        }
        .sheet(isPresented: $showControlLibrary) {
            VStack(spacing: 0) {
                HStack {
                    Text("Control Video Library").font(.headline)
                    Spacer()
                    Button("Done") { showControlLibrary = false }
                }
                .padding()
                ControlLibraryGrid(vm: controlLibraryVM) { item in
                    vm.applyLibraryItem(item)
                    showControlLibrary = false
                }
            }
            .frame(width: 720, height: 520)
        }
    }

    private func showControlVideoPicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select a control video for IC-LoRA"
        if panel.runModal() == .OK, let url = panel.url {
            vm.handleControlVideoDrop(urls: [url])
        }
    }

    private func handleControlVideoDrop(providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
            guard let urlData = data as? Data,
                  let url = URL(dataRepresentation: urlData, relativeTo: nil) else { return }
            let videoTypes = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]
            guard videoTypes.contains(url.pathExtension.lowercased()) else { return }
            DispatchQueue.main.async { vm.handleControlVideoDrop(urls: [url]) }
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

    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.subheadline)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: 0.05)
        }
    }

    private var controlTypeHelp: String {
        switch vm.controlType {
        case .raw:   return "Use the video as-is (for transform IC-LoRAs like colorization or deblur)."
        case .canny: return "Extract edges (canny) on the backend."
        case .pose:  return "Extract a human-pose skeleton on-device (for Union-Control)."
        case .depth: return "Extract a depth map on-device (downloads a small model on first use)."
        }
    }

    private var pipelineDescription: String {
        switch vm.pipelineType {
        case "one-stage":
            return "Dev model + CFG at full resolution (30 steps). No upscale stage."
        case "two-stage":
            return "Dev model + CFG (30 steps), then neural upscale + refinement. Better quality, slower."
        case "two-stage-hq":
            return "HQ res_2s sampler (15 steps) + refinement. Best quality, slowest."
        default:
            return "Distilled model (8+3 steps), half-res + neural upscale. Fastest, good quality."
        }
    }

    // MARK: - Preview Panel

    private var previewPanel: some View {
        Group {
            if let player = player {
                ZStack {
                    PlayerView(player: player)
                        .clipShape(RoundedRectangle(cornerRadius: 12))

                    // Top-right: copy/save/share
                    VStack {
                        HStack {
                            Spacer()
                            if let url = vm.outputVideoURL {
                                videoActionButtons(url: url)
                            }
                        }
                        Spacer()
                        // Retake/extend buttons above the video player controls
                        if vm.outputVideoURL != nil {
                            HStack(spacing: 10) {
                                Button(action: { showRetakeSheet = true }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.triangle.2.circlepath")
                                        Text("Retake")
                                    }
                                    .font(.caption)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)

                                Button(action: { showExtendSheet = true }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.right.to.line")
                                        Text("Extend")
                                    }
                                    .font(.caption)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .padding(.bottom, 52)
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .sheet(isPresented: $showRetakeSheet) {
                    if let url = vm.outputVideoURL {
                        RetakeSheet(
                            sourceVideoPath: url.path,
                            videoDuration: Double(vm.numFrames) / Double(vm.fps),
                            fps: vm.fps
                        )
                        .environmentObject(backendService)
                    }
                }
                .sheet(isPresented: $showExtendSheet) {
                    if let url = vm.outputVideoURL {
                        ExtendSheet(
                            sourceVideoPath: url.path,
                            fps: vm.fps
                        )
                        .environmentObject(backendService)
                    }
                }
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

    // MARK: - Video Action Buttons

    private func videoActionButtons(url: URL) -> some View {
        HStack(spacing: 8) {
            // Copy to clipboard
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.writeObjects([url as NSURL])
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 13))
            }
            .help("Copy video to clipboard")

            // Save to...
            Button {
                let panel = NSSavePanel()
                panel.allowedContentTypes = [.mpeg4Movie]
                panel.nameFieldStringValue = url.lastPathComponent
                if panel.runModal() == .OK, let dest = panel.url {
                    try? FileManager.default.copyItem(at: url, to: dest)
                }
            } label: {
                Image(systemName: "square.and.arrow.down")
                    .font(.system(size: 13))
            }
            .help("Save video as...")

            // Share
            Button {
                let picker = NSSharingServicePicker(items: [url])
                if let view = NSApp.keyWindow?.contentView {
                    picker.show(relativeTo: .zero, of: view, preferredEdge: .minY)
                }
            } label: {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 13))
            }
            .help("Share video")
        }
        .buttonStyle(.borderless)
        .padding(6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(10)
    }
}
