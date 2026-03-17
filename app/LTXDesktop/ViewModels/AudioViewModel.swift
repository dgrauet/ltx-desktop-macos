import Foundation

/// ViewModel for TTS, background music generation, and audio mixing.
@MainActor
class AudioViewModel: ObservableObject {

    // MARK: - TTS State

    @Published var ttsText: String = ""
    @Published var selectedVoice: String = "default"
    @Published var speed: Double = 1.0
    @Published var isGeneratingTTS: Bool = false
    @Published var ttsPath: String?

    // MARK: - Music State

    @Published var selectedGenre: String = "ambient"
    @Published var musicDuration: Double = 10.0
    @Published var isGeneratingMusic: Bool = false
    @Published var musicPath: String?

    // MARK: - Mix State

    @Published var videoAudioVolume: Double = 1.0
    @Published var ttsVolume: Double = 1.0
    @Published var musicVolume: Double = 0.3
    @Published var isMixing: Bool = false
    @Published var mixedOutputPath: String?

    // MARK: - Shared

    @Published var errorMessage: String?

    /// Available TTS voices. IDs must match backend `_VOICE_MAP` keys in `tts_engine.py`.
    static let voices: [(id: String, label: String)] = [
        ("default", "Default"),
        ("kokoro_af_heart", "Heart (F)"),
        ("kokoro_af_bella", "Bella (F)"),
        ("kokoro_af_nova", "Nova (F)"),
        ("kokoro_am_adam", "Adam (M)"),
        ("kokoro_am_echo", "Echo (M)"),
        ("kokoro_bf_alice", "Alice (UK F)"),
        ("kokoro_bm_daniel", "Daniel (UK M)"),
    ]

    /// Available music genres.
    static let genres: [String] = [
        "ambient", "electronic", "orchestral", "jazz",
        "cinematic", "upbeat", "dark", "nature",
    ]

    /// Whether any audio has been generated (TTS or music).
    var hasAnyAudio: Bool {
        ttsPath != nil || musicPath != nil
    }

    /// Whether any audio operation is in progress.
    var isBusy: Bool {
        isGeneratingTTS || isGeneratingMusic || isMixing
    }

    // MARK: - Actions

    func generateTTS(using service: BackendService) async {
        guard !ttsText.isEmpty else {
            errorMessage = "Enter text for voiceover."
            return
        }
        isGeneratingTTS = true
        errorMessage = nil

        do {
            let request = TTSRequest(text: ttsText, voice: selectedVoice, speed: speed)
            let response = try await service.generateTTS(request: request)
            ttsPath = response.outputPath
        } catch {
            errorMessage = "TTS failed: \(error.localizedDescription)"
        }

        isGeneratingTTS = false
    }

    func generateMusic(using service: BackendService) async {
        isGeneratingMusic = true
        errorMessage = nil

        do {
            let request = MusicRequest(genre: selectedGenre, duration: musicDuration)
            let response = try await service.generateMusic(request: request)
            musicPath = response.outputPath
        } catch {
            errorMessage = "Music generation failed: \(error.localizedDescription)"
        }

        isGeneratingMusic = false
    }

    func mixAudio(videoPath: String, using service: BackendService) async {
        isMixing = true
        errorMessage = nil

        do {
            let request = AudioMixRequest(
                videoPath: videoPath,
                ttsPath: ttsPath,
                musicPath: musicPath,
                ttsVolume: ttsVolume,
                musicVolume: musicVolume,
                videoAudioVolume: videoAudioVolume
            )
            let response = try await service.mixAudio(request: request)
            mixedOutputPath = response.outputPath
        } catch {
            errorMessage = "Audio mix failed: \(error.localizedDescription)"
        }

        isMixing = false
    }

    /// Reset all audio state (e.g., when starting a new generation).
    func reset() {
        ttsPath = nil
        musicPath = nil
        mixedOutputPath = nil
        errorMessage = nil
    }
}
