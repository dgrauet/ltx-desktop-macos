import SwiftUI
import AVKit

struct GenerationView: View {
    @EnvironmentObject var backendService: BackendService
    @StateObject private var vm = GenerationViewModel()

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
    }

    // MARK: - Controls Panel

    private var controlsPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
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

                // Generate button
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

    // MARK: - Preview Panel

    private var previewPanel: some View {
        Group {
            if let videoURL = vm.outputVideoURL {
                VideoPlayer(player: AVPlayer(url: videoURL))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
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
