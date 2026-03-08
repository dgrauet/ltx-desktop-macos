import SwiftUI
import AVKit
import UniformTypeIdentifiers

struct GenerationView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = GenerationViewModel()
    @AppStorage("promptEnhanceEnabled") private var enhanceEnabled: Bool = true
    @State private var player: AVPlayer?

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

                Divider()

                // Generate + Preview buttons
                HStack(spacing: 8) {
                    Button(action: {
                        Task { await vm.generate(using: backendService) }
                    }) {
                        HStack {
                            if vm.isGenerating {
                                ProgressView()
                                    .scaleEffect(0.7)
                            }
                            Text(vm.isGenerating ? "Generating..." : "Generate")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(vm.prompt.isEmpty || vm.isGenerating)
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
                    .disabled(vm.prompt.isEmpty || vm.isGenerating)
                    .keyboardShortcut("p", modifiers: .command)
                }

                // Progress
                if vm.isGenerating {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: vm.progress)
                        Text("\(Int(vm.progress * 100))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
