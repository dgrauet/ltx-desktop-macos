import SwiftUI

/// Collapsible audio panel for TTS, music generation, and audio mixing.
/// Displayed in GenerationView after a video has been generated.
struct AudioPanelView: View {
    @EnvironmentObject var backendService: BackendService
    @ObservedObject var vm: AudioViewModel
    let videoPath: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Audio", systemImage: "waveform")
                .font(.headline)

            // TTS Section
            ttsSection

            // Music Section
            musicSection

            // Mix Section (only when video + at least one audio track)
            if vm.hasAnyAudio {
                mixSection
            }

            // Error
            if let error = vm.errorMessage {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
    }

    // MARK: - TTS Section

    private var ttsSection: some View {
        DisclosureGroup("Voiceover (TTS)") {
            VStack(alignment: .leading, spacing: 8) {
                TextEditor(text: $vm.ttsText)
                    .frame(minHeight: 60, maxHeight: 100)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .background(Color(.textBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )

                HStack {
                    Text("Voice")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.selectedVoice) {
                        ForEach(AudioViewModel.voices, id: \.id) { voice in
                            Text(voice.label).tag(voice.id)
                        }
                    }
                    .frame(width: 140)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Speed")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.1f\u{00D7}", vm.speed))
                            .font(.subheadline)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $vm.speed, in: 0.5...2.0, step: 0.1)
                }

                Button(action: {
                    Task { await vm.generateTTS(using: backendService) }
                }) {
                    HStack {
                        if vm.isGeneratingTTS {
                            ProgressView()
                                .scaleEffect(0.6)
                        }
                        Text(vm.isGeneratingTTS ? "Generating..." : "Generate Voiceover")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(vm.ttsText.isEmpty || vm.isGeneratingTTS)

                if let path = vm.ttsPath {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                        Text(URL(fileURLWithPath: path).lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.top, 4)
        }
    }

    // MARK: - Music Section

    private var musicSection: some View {
        DisclosureGroup("Background Music") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Genre")
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: $vm.selectedGenre) {
                        ForEach(AudioViewModel.genres, id: \.self) { genre in
                            Text(genre.capitalized).tag(genre)
                        }
                    }
                    .frame(width: 140)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Duration")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.0fs", vm.musicDuration))
                            .font(.subheadline)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $vm.musicDuration, in: 1...60, step: 1)
                }

                Button(action: {
                    Task { await vm.generateMusic(using: backendService) }
                }) {
                    HStack {
                        if vm.isGeneratingMusic {
                            ProgressView()
                                .scaleEffect(0.6)
                        }
                        Text(vm.isGeneratingMusic ? "Generating..." : "Generate Music")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(vm.isGeneratingMusic)

                if let path = vm.musicPath {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                        Text(URL(fileURLWithPath: path).lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.top, 4)
        }
    }

    // MARK: - Mix Section

    private var mixSection: some View {
        DisclosureGroup("Mix Audio") {
            VStack(alignment: .leading, spacing: 8) {
                volumeSlider(label: "Video Audio", value: $vm.videoAudioVolume)

                if vm.ttsPath != nil {
                    volumeSlider(label: "Voiceover", value: $vm.ttsVolume)
                }

                if vm.musicPath != nil {
                    volumeSlider(label: "Music", value: $vm.musicVolume)
                }

                Button(action: {
                    Task { await vm.mixAudio(videoPath: videoPath, using: backendService) }
                }) {
                    HStack {
                        if vm.isMixing {
                            ProgressView()
                                .scaleEffect(0.6)
                        }
                        Text(vm.isMixing ? "Mixing..." : "Mix Audio")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isMixing)

                if let path = vm.mixedOutputPath {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                        Text(URL(fileURLWithPath: path).lastPathComponent)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.top, 4)
        }
    }

    // MARK: - Helpers

    private func volumeSlider(label: String, value: Binding<Double>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.subheadline)
                Spacer()
                Text(String(format: "%.0f%%", value.wrappedValue * 100))
                    .font(.subheadline)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: 0.0...1.0, step: 0.05)
        }
    }
}
